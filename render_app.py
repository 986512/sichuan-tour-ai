# ==================== KG-BiVAF 推理服务 - Render 部署优化版 ====================
# 专门为 Render.com 平台优化的部署版本
# 特点：更轻量、启动更快、自动健康检查

import os
import sys
import warnings
import re
import pickle
import json
import time
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 禁用警告
warnings.filterwarnings('ignore')

# 环境变量配置
PORT = int(os.environ.get('PORT', 8000))
MODEL_DIR = os.environ.get('MODEL_DIR', './output')

# 第三方库导入
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import jieba

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"推理服务启动，使用设备: {device}")

# ==================== 模型定义（与训练时保持一致） ====================
class TextCNN(nn.Module):
    """TextCNN 情感分析模型"""
    def __init__(self, vocab_size, embedding_dim=200, num_filters=100,
                 filter_sizes=[2,3,4], num_classes=7, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k, padding=k-1)
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        conv_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outs.append(pooled)
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class BiVAF(nn.Module):
    """BiVAF 推荐模型"""
    def __init__(self, user_dim, item_dim, latent_dim=64, dropout=0.2):
        super().__init__()
        self.user_encoder = nn.Sequential(
            nn.Linear(user_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout)
        )
        self.user_mu = nn.Linear(128, latent_dim)
        self.user_logvar = nn.Linear(128, latent_dim)
        self.item_encoder = nn.Sequential(
            nn.Linear(item_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout)
        )
        self.item_mu = nn.Linear(128, latent_dim)
        self.item_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim*2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def encode_user(self, x):
        h = self.user_encoder(x)
        mu = self.user_mu(h)
        logvar = self.user_logvar(h)
        return mu, logvar

    def encode_item(self, x):
        h = self.item_encoder(x)
        mu = self.item_mu(h)
        logvar = self.item_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, user_feat, item_feat):
        user_mu, user_logvar = self.encode_user(user_feat)
        item_mu, item_logvar = self.encode_item(item_feat)
        user_latent = self.reparameterize(user_mu, user_logvar)
        item_latent = self.reparameterize(item_mu, item_logvar)
        z = torch.cat([user_latent, item_latent], dim=1)
        pred = self.decoder(z).view(-1)
        return pred, (user_mu, user_logvar, item_mu, item_logvar)


# ==================== 全局变量 ====================
model = None
attraction_features = None
attraction_df = None
scaler_text = None
lda_model = None
vectorizer = None
vocab = None
textcnn = None
tokenizer = None
bert_model = None
cold_start_sim_matrix = None
attraction_id_to_idx = {}

model_loaded = False
load_error = None

# ==================== 工具函数 ====================
def clean_text(text: str) -> str:
    """清洗文本"""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）【】《》\s]', '', text)
    return text.strip()


def extract_simple_features(text: str) -> np.ndarray:
    """简化版特征提取（不使用BERT，适合轻量部署）"""
    text = clean_text(text)
    if len(text) == 0:
        # 返回零向量
        if scaler_text:
            return np.zeros(scaler_text.n_features_in_)
        else:
            return np.zeros(15 + 7)  # LDA + 情感

    features_list = []

    # LDA 主题分布
    if lda_model and vectorizer:
        try:
            seg_text = ' '.join([w for w in jieba.lcut(text) if len(w) > 1])
            bow = vectorizer.transform([seg_text])
            lda_dist = lda_model.transform(bow)[0]
            features_list.append(lda_dist)
        except Exception as e:
            logger.warning(f"LDA 特征提取失败: {e}")
            features_list.append(np.zeros(15))

    # TextCNN 情感特征
    if textcnn and vocab:
        try:
            words = jieba.lcut(text)
            seq = [vocab.get(w, 0) for w in words[:100]]
            if len(seq) < 100:
                seq += [0] * (100 - len(seq))
            input_tensor = torch.LongTensor([seq]).to(device)
            with torch.no_grad():
                senti = textcnn(input_tensor).cpu().numpy()[0]
            features_list.append(senti)
        except Exception as e:
            logger.warning(f"TextCNN 情感特征提取失败: {e}")
            features_list.append(np.zeros(7))

    # 融合特征
    if len(features_list) == 0:
        return np.zeros(15 + 7)

    feat = np.concatenate(features_list)

    # 标准化
    if scaler_text:
        try:
            feat_scaled = scaler_text.transform(feat.reshape(1, -1))[0]
        except:
            feat_scaled = feat
    else:
        feat_scaled = feat

    return feat_scaled


def recommend_for_query(
    query_text: str,
    top_k: int = 10,
    category_filter: Optional[str] = None,
    max_price: Optional[float] = None
) -> List[Dict[str, Any]]:
    """基于查询文本进行推荐"""
    if not model_loaded:
        raise RuntimeError(f"模型未加载完成: {load_error}")

    # 提取用户特征
    user_feat = extract_simple_features(query_text)

    # 批量预测
    user_tensor = torch.FloatTensor(user_feat).to(device)
    n_items = len(attraction_df)
    user_expanded = user_tensor.expand(n_items, -1)

    with torch.no_grad():
        pred, _ = model(
            user_expanded,
            torch.FloatTensor(attraction_features).to(device)
        )
        scores = pred.cpu().numpy().flatten()

    # 获取 Top-K 推荐
    top_indices = scores.argsort()[::-1][:top_k * 3]

    # 构建推荐结果
    recommendations = []
    for idx in top_indices:
        attraction_id = attraction_df.iloc[idx]['attraction_id']
        category = attraction_df.iloc[idx]['category']
        score = scores[idx]

        # 应用过滤条件
        if category_filter and category != category_filter:
            continue
        if max_price and attraction_df.iloc[idx].get('ticket', 0) > max_price:
            continue

        recommendations.append({
            'attraction_id': attraction_id,
            'category': category,
            'score': float(score),
            'rank': len(recommendations) + 1
        })

        if len(recommendations) >= top_k:
            break

    return recommendations


def recommend_similar_attractions(
    target_attraction_id: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """相似景点推荐（冷启动）"""
    if cold_start_sim_matrix is None:
        raise RuntimeError("冷启动相似度矩阵未加载")

    if target_attraction_id not in attraction_id_to_idx:
        raise ValueError(f"景点ID不存在: {target_attraction_id}")

    target_idx = attraction_id_to_idx[target_attraction_id]
    sim_scores = cold_start_sim_matrix[target_idx]

    # 获取最相似的景点（排除自己）
    top_indices = sim_scores.argsort()[::-1][1:top_k+1]

    recommendations = []
    for idx in top_indices:
        attraction_id = attraction_df.iloc[idx]['attraction_id']
        category = attraction_df.iloc[idx]['category']
        similarity = float(sim_scores[idx])

        recommendations.append({
            'attraction_id': attraction_id,
            'category': category,
            'similarity': similarity,
            'rank': len(recommendations) + 1
        })

    return recommendations


# ==================== 模型加载函数 ====================
def load_models():
    """加载所有模型和预处理组件"""
    global model, attraction_features, attraction_df, scaler_text, lda_model
    global vectorizer, vocab, textcnn, tokenizer, bert_model
    global cold_start_sim_matrix, attraction_id_to_idx
    global model_loaded, load_error

    try:
        logger.info("=" * 60)
        logger.info("开始加载模型和组件...")
        start_time = time.time()

        # 检查模型目录
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"模型目录不存在: {MODEL_DIR}")

        logger.info(f"模型目录: {MODEL_DIR}")

        # 1. 加载景区特征
        attraction_features_path = os.path.join(MODEL_DIR, 'attraction_features.npy')
        if not os.path.exists(attraction_features_path):
            raise FileNotFoundError(f"景区特征文件不存在: {attraction_features_path}")
        attraction_features = np.load(attraction_features_path)
        logger.info(f"✓ 景区特征加载完成，维度: {attraction_features.shape}")

        # 2. 加载景区信息
        attraction_info_path = os.path.join(MODEL_DIR, 'attraction_info.csv')
        if not os.path.exists(attraction_info_path):
            raise FileNotFoundError(f"景区信息文件不存在: {attraction_info_path}")
        attraction_df = pd.read_csv(attraction_info_path)
        logger.info(f"✓ 景区信息加载完成，数量: {len(attraction_df)}")

        # 构建景区ID到索引的映射
        attraction_id_to_idx = {aid: i for i, aid in enumerate(attraction_df['attraction_id'].tolist())}

        # 3. 加载用户特征（用于获取维度）
        user_features_path = os.path.join(MODEL_DIR, 'user_features.npy')
        if os.path.exists(user_features_path):
            user_features = np.load(user_features_path)
            user_dim = user_features.shape[1]
            logger.info(f"✓ 用户特征加载完成，维度: {user_features.shape}")
        else:
            logger.warning("用户特征文件不存在，使用景区特征维度作为用户维度")
            user_dim = attraction_features.shape[1]

        item_dim = attraction_features.shape[1]

        # 4. 加载 BiVAF 模型
        model_path = os.path.join(MODEL_DIR, 'kg_bivaf.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BiVAF 模型文件不存在: {model_path}")

        model = BiVAF(
            user_dim,
            item_dim,
            latent_dim=128,  # 与训练时一致
            dropout=0.1  # 与训练时一致
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"✓ BiVAF 模型加载完成")

        # 5. 加载文本预处理组件
        scaler_text_path = os.path.join(MODEL_DIR, 'scaler_text.pkl')
        if os.path.exists(scaler_text_path):
            with open(scaler_text_path, 'rb') as f:
                scaler_text = pickle.load(f)
            logger.info(f"✓ 文本标准化器加载完成")

        lda_path = os.path.join(MODEL_DIR, 'lda_model.pkl')
        if os.path.exists(lda_path):
            with open(lda_path, 'rb') as f:
                lda_model = pickle.load(f)
            logger.info(f"✓ LDA 模型加载完成")

        vectorizer_path = os.path.join(MODEL_DIR, 'vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            logger.info(f"✓ 向量化器加载完成")

        vocab_path = os.path.join(MODEL_DIR, 'vocab.pkl')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            logger.info(f"✓ 词表加载完成，大小: {len(vocab)}")

        # 6. 加载 TextCNN 模型
        textcnn_path = os.path.join(MODEL_DIR, 'textcnn.pth')
        if os.path.exists(textcnn_path) and vocab is not None:
            textcnn = TextCNN(len(vocab)).to(device)
            textcnn.load_state_dict(torch.load(textcnn_path, map_location=device))
            textcnn.eval()
            logger.info(f"✓ TextCNN 模型加载完成")

        # 7. 加载冷启动相似度矩阵
        sim_matrix_path = os.path.join(MODEL_DIR, 'cold_start_sim_matrix.npy')
        if os.path.exists(sim_matrix_path):
            cold_start_sim_matrix = np.load(sim_matrix_path)
            logger.info(f"✓ 冷启动相似度矩阵加载完成")

        model_loaded = True
        load_time = time.time() - start_time
        logger.info(f"\n✓✓✓ 所有模型加载完成！总耗时: {load_time:.2f}秒")
        logger.info("=" * 60)

    except Exception as e:
        model_loaded = False
        load_error = str(e)
        logger.error(f"✗✗✗ 模型加载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="KG-BiVAF 景区推荐API",
    description="基于知识图谱和 BiVAF 的四川省研学旅游景区推荐服务",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 数据模型 ====================
class RecommendRequest(BaseModel):
    query_text: str = Field(..., description="用户查询文本")
    top_k: int = Field(default=10, ge=1, le=50, description="返回推荐数量")
    category_filter: Optional[str] = Field(None, description="类别过滤")
    max_price: Optional[float] = Field(None, ge=0, description="最高票价")


class SimilarRequest(BaseModel):
    target_attraction_id: str = Field(..., description="目标景点ID")
    top_k: int = Field(default=5, ge=1, le=20, description="返回相似景点数量")


class RecommendResponse(BaseModel):
    success: bool
    message: str
    recommendations: List[Dict[str, Any]]
    query_text: str
    timestamp: str


class SimilarResponse(BaseModel):
    success: bool
    message: str
    recommendations: List[Dict[str, Any]]
    target_attraction_id: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_attractions: int
    load_error: Optional[str]


# ==================== 启动时加载模型 ====================
@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    load_models()


# ==================== API 端点 ====================
@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "KG-BiVAF 景区推荐API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_loaded,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if model_loaded else "loading",
        model_loaded=model_loaded,
        device=str(device),
        num_attractions=len(attraction_df) if attraction_df is not None else 0,
        load_error=load_error
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """推荐接口"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail=f"模型未加载完成: {load_error}")

    try:
        recommendations = recommend_for_query(
            query_text=request.query_text,
            top_k=request.top_k,
            category_filter=request.category_filter,
            max_price=request.max_price
        )

        return RecommendResponse(
            success=True,
            message=f"成功推荐 {len(recommendations)} 个景区",
            recommendations=recommendations,
            query_text=request.query_text,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"推荐失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar", response_model=SimilarResponse)
async def get_similar(request: SimilarRequest):
    """相似景点推荐"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail=f"模型未加载完成: {load_error}")

    try:
        recommendations = recommend_similar_attractions(
            target_attraction_id=request.target_attraction_id,
            top_k=request.top_k
        )

        return SimilarResponse(
            success=True,
            message=f"成功找到 {len(recommendations)} 个相似景点",
            recommendations=recommendations,
            target_attraction_id=request.target_attraction_id,
            timestamp=datetime.now().isoformat()
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"相似推荐失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/attractions")
async def get_attractions(category: Optional[str] = None, limit: int = 100):
    """获取景区列表"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载完成")

    try:
        df = attraction_df.copy()

        if category:
            df = df[df['category'] == category]

        df = df.head(limit)

        attractions = df.to_dict('records')

        return {
            "success": True,
            "attractions": attractions,
            "total": len(attractions)
        }

    except Exception as e:
        logger.error(f"获取景点列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories")
async def get_categories():
    """获取所有景区类别"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载完成")

    try:
        categories = attraction_df['category'].unique().tolist()
        return {
            "success": True,
            "categories": categories,
            "total": len(categories)
        }

    except Exception as e:
        logger.error(f"获取类别失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 本地测试入口 ====================
if __name__ == "__main__":
    import uvicorn

    # 先加载模型
    load_models()

    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )
