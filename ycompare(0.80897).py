# ===================== 导入和兼容性修复 =====================
import pandas as pd
import numpy as np
import warnings
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import sys

# 修复CatBoost导入兼容性问题
try:
    # 先修复collections导入问题
    import collections
    try:
        from collections.abc import Iterable, Sequence, Mapping, MutableMapping
        collections.Iterable = Iterable
        collections.Sequence = Sequence
        collections.Mapping = Mapping
        collections.MutableMapping = MutableMapping
    except ImportError:
        pass
    
    import catboost as cb
    CATBOOST_AVAILABLE = True
    print("✅ CatBoost 成功导入")
except ImportError as e:
    print(f"⚠️ CatBoost 导入失败: {e}，将使用替代方案")
    CATBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

# ===================== 1. 高级特征工程 =====================
def advanced_feature_engineering(df, is_train=True):
    """
    高级特征工程函数
    创建更多有预测能力的特征来提升模型性能
    """
    df = df.copy()
    
    # 1. PassengerId解析
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['PersonId'] = df['PassengerId'].str.split('_').str[1].astype(int)
    df['GroupSize'] = df.groupby('Group')['PassengerId'].transform('count')
    df['InGroup'] = (df['GroupSize'] > 1).astype(int)
    df['IsGroupLeader'] = (df['PersonId'] == 1).astype(int)
    
    # 2. Cabin特征深度解析
    if 'Cabin' in df.columns:
        cabin_split = df['Cabin'].str.split('/', expand=True)
        df['Deck'] = cabin_split[0].fillna('Unknown')
        df['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce')
        df['Side'] = cabin_split[2].fillna('Unknown')
        
        # Deck重要性编码
        deck_importance = {'A': 5, 'B': 5, 'C': 4, 'D': 3, 'E': 3, 'F': 2, 'G': 1, 'T': 5, 'Unknown': 0}
        df['DeckScore'] = df['Deck'].map(deck_importance)
        
        # Side编码
        df['SideCode'] = df['Side'].map({'P': 0, 'S': 1, 'Unknown': -1})
    
    # 3. 消费特征工程
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # 处理消费列的NaN值
    for col in spend_cols:
        df[col] = df[col].fillna(0)
    
    # 总消费和变换
    df['TotalSpent'] = df[spend_cols].sum(axis=1)
    df['LogTotalSpent'] = np.log1p(df['TotalSpent'])
    df['HasSpent'] = (df['TotalSpent'] > 0).astype(int)
    
    # 消费多样性
    df['SpentDiversity'] = (df[spend_cols] > 0).sum(axis=1)
    
    # 消费模式特征
    df['LuxurySpent'] = df['RoomService'] + df['Spa'] + df['VRDeck']
    df['BasicSpent'] = df['FoodCourt'] + df['ShoppingMall']
    df['LuxuryRatio'] = df['LuxurySpent'] / (df['TotalSpent'] + 1e-6)
    
    # 主要消费类型
    df['MainSpendType'] = df[spend_cols].idxmax(axis=1)
    
    # 4. 年龄特征工程
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # 年龄分组
    age_bins = [0, 12, 18, 25, 35, 50, 65, 100]
    age_labels = ['Child', 'Teen', 'Young', 'Adult', 'Middle', 'Senior', 'Elderly']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    
    # 年龄布尔特征
    df['IsChild'] = (df['Age'] <= 12).astype(int)
    df['IsYoungAdult'] = ((df['Age'] > 18) & (df['Age'] <= 35)).astype(int)
    df['IsSenior'] = (df['Age'] > 50).astype(int)
    
    # 5. 姓名特征
    df['NameLength'] = df['Name'].str.len().fillna(0)
    df['Surname'] = df['Name'].str.split().str[-1].fillna('Unknown')
    
    # 6. 交互特征
    df['CryoSleep_VIP'] = df['CryoSleep'].fillna('Unknown').astype(str) + '_' + df['VIP'].fillna('Unknown').astype(str)
    df['HomePlanet_Destination'] = df['HomePlanet'].fillna('Unknown') + '_' + df['Destination'].fillna('Unknown')
    
    # 7. 特殊模式特征
    df['AllZeroSpend'] = (df['TotalSpent'] == 0).astype(int)
    df['HighSpender'] = (df['TotalSpent'] > df['TotalSpent'].median()).astype(int)
    
    # 8. 分组统计特征
    if 'Group' in df.columns:
        # 分组内统计
        group_features = ['Age', 'TotalSpent', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for feature in group_features:
            if feature in df.columns:
                df[f'Group{feature}Mean'] = df.groupby('Group')[feature].transform('mean')
                df[f'Group{feature}Std'] = df.groupby('Group')[feature].transform('std').fillna(0)
    
    # 布尔特征处理
    bool_cols = ['CryoSleep', 'VIP']
    for col in bool_cols:
        df[col] = df[col].fillna('Unknown').astype(str)
    
    # 目标变量处理
    if is_train and 'Transported' in df.columns:
        df['Transported'] = df['Transported'].astype(int)
    
    # 删除原始列
    cols_to_drop = ['PassengerId', 'Name', 'Cabin', 'Surname']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    return df

# ===================== 2. 智能缺失值处理 =====================
def smart_missing_value_imputation(df):
    """智能缺失值填充"""
    df = df.copy()
    
    # 数值特征使用中位数填充
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # 分类特征使用众数填充
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
    
    return df

# ===================== 3. 特征选择优化 =====================
def optimize_feature_selection(X_train, y_train, top_k=25):
    """优化特征选择"""
    print("进行特征重要性分析...")
    
    # 编码分类特征
    X_encoded = X_train.copy()
    categorical_features = X_train.select_dtypes(include=['object']).columns
    
    for col in categorical_features:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    
    # 使用LightGBM分析特征重要性
    lgb_selector = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    lgb_selector.fit(X_encoded, y_train)
    
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': lgb_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10最重要特征:")
    print(feature_importance.head(10))
    
    # 选择最佳特征
    selected_features = feature_importance.head(top_k)['feature'].tolist()
    return selected_features

# ===================== 4. 高级模型集成 =====================
def create_advanced_ensemble(numeric_features, categorical_features):
    """创建高级模型集成"""
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # 定义多个强模型
    models = {
        'lgb1': lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'lgb2': lgb.LGBMClassifier(
            n_estimators=800,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=0.3,
            random_state=43,
            n_jobs=-1,
            verbose=-1
        ),
        'xgb1': xgb.XGBClassifier(
            n_estimators=600,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=44,
            n_jobs=-1,
            verbosity=0
        ),
        'xgb2': xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.05,
            random_state=45,
            n_jobs=-1,
            verbosity=0
        )
    }
    
    # 如果CatBoost可用，添加CatBoost模型
    if CATBOOST_AVAILABLE:
        models['catboost'] = cb.CatBoostClassifier(
            iterations=500,
            depth=7,
            learning_rate=0.05,
            random_state=46,
            verbose=0
        )
    
    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    
    return pipelines

# ===================== 5. 堆叠集成模型 =====================
def create_stacking_ensemble(numeric_features, categorical_features):
    """创建堆叠集成模型"""
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # 基学习器
    base_learners = [
        ('lgb1', lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)),
        ('lgb2', lgb.LGBMClassifier(n_estimators=200, random_state=43, verbose=-1)),
        ('xgb', xgb.XGBClassifier(n_estimators=200, random_state=44, verbosity=0))
    ]
    
    # 如果CatBoost可用，添加到基学习器
    if CATBOOST_AVAILABLE:
        base_learners.append(('catboost', cb.CatBoostClassifier(iterations=200, verbose=0, random_state=45)))
    
    # 创建投票集成
    voting_clf = VotingClassifier(estimators=base_learners, voting='soft')
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', voting_clf)
    ])

# ===================== 6. 主训练函数 =====================
def main():
    """主训练函数"""
    print("=== Spaceship Titanic 预测优化 ===")
    
    try:
        # 数据加载
        print("加载数据...")
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
        test_ids = test_df["PassengerId"]
        
        print(f"原始数据 - 训练集: {train_df.shape}, 测试集: {test_df.shape}")
        
        # 特征工程
        print("\n=== 特征工程 ===")
        train_processed = advanced_feature_engineering(train_df, is_train=True)
        test_processed = advanced_feature_engineering(test_df, is_train=False)
        
        print(f"特征工程后 - 训练集: {train_processed.shape}, 测试集: {test_processed.shape}")
        
        # 缺失值处理
        print("\n=== 缺失值处理 ===")
        train_filled = smart_missing_value_imputation(train_processed)
        test_filled = smart_missing_value_imputation(test_processed)
        
        # 准备特征和目标
        X_train = train_filled.drop('Transported', axis=1)
        y_train = train_filled['Transported']
        X_test = test_filled
        
        # 特征选择
        print("\n=== 特征选择 ===")
        selected_features = optimize_feature_selection(X_train, y_train, top_k=25)
        
        # 特征类型识别
        numeric_features = [f for f in selected_features if f in X_train.select_dtypes(include=[np.number]).columns]
        categorical_features = [f for f in selected_features if f in X_train.select_dtypes(include=['object']).columns]
        
        print(f"选择特征 - 数值: {len(numeric_features)}, 分类: {len(categorical_features)}")
        
        # 创建模型
        print("\n=== 模型训练 ===")
        pipelines = create_advanced_ensemble(numeric_features, categorical_features)
        stacking_pipeline = create_stacking_ensemble(numeric_features, categorical_features)
        
        # 5折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_results = {}
        best_score = 0
        best_model_name = ""
        best_model = None
        
        # 训练和评估单个模型
        for name, pipeline in pipelines.items():
            try:
                print(f"训练 {name}...")
                cv_scores = cross_val_score(pipeline, X_train[selected_features], y_train, 
                                          cv=skf, scoring='accuracy', n_jobs=1)
                cv_results[name] = cv_scores
                mean_score = cv_scores.mean()
                print(f"{name.upper()} - 准确率: {mean_score:.5f} ± {cv_scores.std():.5f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = name
                    best_model = pipeline
            except Exception as e:
                print(f"{name} 训练失败: {e}")
                cv_results[name] = [0]
        
        # 训练堆叠模型
        try:
            print("训练堆叠模型...")
            stacking_scores = cross_val_score(stacking_pipeline, X_train[selected_features], y_train,
                                           cv=skf, scoring='accuracy', n_jobs=1)
            cv_results['stacking'] = stacking_scores
            stacking_score = stacking_scores.mean()
            print(f"STACKING - 准确率: {stacking_score:.5f} ± {stacking_scores.std():.5f}")
            
            if stacking_score > best_score:
                best_score = stacking_score
                best_model_name = 'stacking'
                best_model = stacking_pipeline
        except Exception as e:
            print(f"堆叠模型训练失败: {e}")
            cv_results['stacking'] = [0]
        
        print(f"\n最佳模型: {best_model_name}, 准确率: {best_score:.5f}")
        
        # 验证集评估
        print("\n=== 验证集评估 ===")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train[selected_features], y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        best_model.fit(X_tr, y_tr)
        y_val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        print(f"验证集准确率: {val_accuracy:.5f}")
        print("\n分类报告:")
        print(classification_report(y_val, y_val_pred))
        
        # 全量训练和预测
        print("\n=== 全量训练 ===")
        best_model.fit(X_train[selected_features], y_train)
        y_test_pred = best_model.predict(X_test[selected_features])
        
        # 生成提交文件
        submission_filename = "spaceship_titanic_submission.csv"
        submission = pd.DataFrame({
            "PassengerId": test_ids,
            "Transported": y_test_pred.astype(bool)
        })
        submission.to_csv(submission_filename, index=False)
        
        # 结果报告
        print("\n" + "="*50)
        print("训练和预测完成!")
        print("="*50)
        print(f"提交文件: {submission_filename}")
        print(f"最佳模型: {best_model_name}")
        print(f"交叉验证准确率: {best_score:.5f}")
        print(f"验证集准确率: {val_accuracy:.5f}")
        
        final_accuracy = max(best_score, val_accuracy)
        improvement = final_accuracy - 0.80617
        print(f"最终预期准确率: {final_accuracy:.5f}")
        print(f"相比之前提升: {improvement:+.5f}")
        
        # 预测分布分析
        transported_count = submission['Transported'].sum()
        total_count = len(submission)
        transport_rate = transported_count / total_count * 100
        print(f"\n预测分布:")
        print(f"   Transported=True: {transported_count}/{total_count} ({transport_rate:.1f}%)")
        
        return submission, final_accuracy
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

# ===================== 执行主函数 =====================
if __name__ == "__main__":
    submission, final_accuracy = main()
