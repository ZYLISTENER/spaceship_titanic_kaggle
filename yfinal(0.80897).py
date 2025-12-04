"""
Spaceship Titanic 乘客运输预测模型
使用 LightGBM 进行分类预测，包含特征工程、特征选择和模型训练
"""

# 导入标准库
import warnings
import os
import sys

# 导入数据处理库
import pandas as pd
import numpy as np

# 导入机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 导入 LightGBM
import lightgbm as lgb

# 忽略警告信息
warnings.filterwarnings('ignore')


def advanced_feature_engineering(df, is_train=True):
    """
    高级特征工程函数
    
    参数:
    ----------
    df : pandas.DataFrame
        输入数据框
    is_train : bool, 默认=True
        是否为训练数据（决定是否处理目标列）
    
    返回:
    ----------
    pandas.DataFrame
        处理后的数据框
    """
    df = df.copy()
    
    # 1. 从PassengerId提取组信息
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['PersonId'] = df['PassengerId'].str.split('_').str[1].astype(int)
    df['GroupSize'] = df.groupby('Group')['PassengerId'].transform('count')
    df['InGroup'] = (df['GroupSize'] > 1).astype(int)
    df['IsGroupLeader'] = (df['PersonId'] == 1).astype(int)
    
    # 2. 舱位信息处理
    if 'Cabin' in df.columns:
        cabin_split = df['Cabin'].str.split('/', expand=True)
        df['Deck'] = cabin_split[0].fillna('Unknown')
        df['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce')
        df['Side'] = cabin_split[2].fillna('Unknown')
        
        # 甲板重要性评分
        deck_importance = {'A': 5, 'B': 5, 'C': 4, 'D': 3, 'E': 3, 'F': 2, 'G': 1, 'T': 5, 'Unknown': 0}
        df['DeckScore'] = df['Deck'].map(deck_importance)
        
        # 船舷编码
        df['SideCode'] = df['Side'].map({'P': 0, 'S': 1, 'Unknown': -1})
    
    # 3. 消费特征处理
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # 填充缺失值为0
    for col in spend_cols:
        df[col] = df[col].fillna(0)
    
    # 创建消费相关特征
    df['TotalSpent'] = df[spend_cols].sum(axis=1)
    df['LogTotalSpent'] = np.log1p(df['TotalSpent'])
    df['HasSpent'] = (df['TotalSpent'] > 0).astype(int)
    df['SpentDiversity'] = (df[spend_cols] > 0).sum(axis=1)
    
    # 奢侈消费与基本消费
    df['LuxurySpent'] = df['RoomService'] + df['Spa'] + df['VRDeck']
    df['BasicSpent'] = df['FoodCourt'] + df['ShoppingMall']
    df['LuxuryRatio'] = df['LuxurySpent'] / (df['TotalSpent'] + 1e-6)
    df['MainSpendType'] = df[spend_cols].idxmax(axis=1)
    
    # 4. 年龄特征处理
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # 年龄分组
    age_bins = [0, 12, 18, 25, 35, 50, 65, 100]
    age_labels = ['Child', 'Teen', 'Young', 'Adult', 'Middle', 'Senior', 'Elderly']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    
    # 特殊年龄段标识
    df['IsChild'] = (df['Age'] <= 12).astype(int)
    df['IsYoungAdult'] = ((df['Age'] > 18) & (df['Age'] <= 35)).astype(int)
    df['IsSenior'] = (df['Age'] > 50).astype(int)
    
    # 5. 姓名特征处理
    df['NameLength'] = df['Name'].str.len().fillna(0)
    df['Surname'] = df['Name'].str.split().str[-1].fillna('Unknown')
    
    # 6. 组合特征
    df['CryoSleep_VIP'] = df['CryoSleep'].fillna('Unknown').astype(str) + '_' + df['VIP'].fillna('Unknown').astype(str)
    df['HomePlanet_Destination'] = df['HomePlanet'].fillna('Unknown') + '_' + df['Destination'].fillna('Unknown')
    
    # 7. 消费行为特征
    df['AllZeroSpend'] = (df['TotalSpent'] == 0).astype(int)
    df['HighSpender'] = (df['TotalSpent'] > df['TotalSpent'].median()).astype(int)
    
    # 8. 组级别聚合特征
    if 'Group' in df.columns:
        group_features = ['Age', 'TotalSpent', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for feature in group_features:
            if feature in df.columns:
                df[f'Group{feature}Mean'] = df.groupby('Group')[feature].transform('mean')
                df[f'Group{feature}Std'] = df.groupby('Group')[feature].transform('std').fillna(0)
    
    # 9. 布尔列处理
    bool_cols = ['CryoSleep', 'VIP']
    for col in bool_cols:
        df[col] = df[col].fillna('Unknown').astype(str)
    
    # 10. 目标变量转换（仅训练集）
    if is_train and 'Transported' in df.columns:
        df['Transported'] = df['Transported'].astype(int)
    
    # 11. 删除无用列
    cols_to_drop = ['PassengerId', 'Name', 'Cabin', 'Surname']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    return df


def smart_missing_value_imputation(df):
    """
    智能缺失值填充
    
    参数:
    ----------
    df : pandas.DataFrame
        输入数据框
    
    返回:
    ----------
    pandas.DataFrame
        填充后的数据框
    """
    df = df.copy()
    
    # 数值型列：用中位数填充
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # 分类型列：用众数填充
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
    
    return df


def optimize_feature_selection(X_train, y_train, top_k=25):
    """
    基于特征重要性进行特征选择
    
    参数:
    ----------
    X_train : pandas.DataFrame
        训练特征
    y_train : pandas.Series
        训练标签
    top_k : int, 默认=25
        选择的最重要特征数量
    
    返回:
    ----------
    list
        选择的特征列表
    """
    print("进行特征重要性分析...")
    
    # 编码分类特征
    X_encoded = X_train.copy()
    categorical_features = X_train.select_dtypes(include=['object']).columns
    
    for col in categorical_features:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    
    # 使用LightGBM计算特征重要性
    lgb_selector = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    lgb_selector.fit(X_encoded, y_train)
    
    # 特征重要性排序
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': lgb_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10最重要特征:")
    print(feature_importance.head(10))
    
    # 选择最重要的top_k个特征
    selected_features = feature_importance.head(top_k)['feature'].tolist()
    return selected_features


def create_lgb1_model(numeric_features, categorical_features):
    """
    创建LightGBM模型管道
    
    参数:
    ----------
    numeric_features : list
        数值特征列表
    categorical_features : list
        分类特征列表
    
    返回:
    ----------
    sklearn.pipeline.Pipeline
        完整的模型管道
    """
    # 创建预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),  # 数值特征标准化
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # 分类特征独热编码
        ])
    
    # 创建LightGBM分类器
    lgb1_model = lgb.LGBMClassifier(
        n_estimators=1000,      # 树的数量
        max_depth=8,            # 树的最大深度
        learning_rate=0.02,     # 学习率
        subsample=0.8,          # 行采样比例
        colsample_bytree=0.8,   # 列采样比例
        reg_alpha=0.1,          # L1正则化
        reg_lambda=0.2,         # L2正则化
        random_state=42,        # 随机种子
        n_jobs=-1,              # 使用所有CPU核心
        verbose=-1              # 不显示训练日志
    )
    
    # 创建管道
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # 数据预处理
        ('classifier', lgb1_model)       # 模型训练
    ])
    
    return pipeline


def main():
    """
    主函数：控制整个机器学习流水线
    """
    print("=== Spaceship Titanic 预测优化 ===")
    print("使用LGB1模型")
    
    try:
        # ========== 1. 数据加载 ==========
        print("加载数据...")
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
        test_ids = test_df["PassengerId"]
        
        print(f"原始数据 - 训练集: {train_df.shape}, 测试集: {test_df.shape}")
        
        # ========== 2. 特征工程 ==========
        print("\n=== 特征工程 ===")
        train_processed = advanced_feature_engineering(train_df, is_train=True)
        test_processed = advanced_feature_engineering(test_df, is_train=False)
        
        print(f"特征工程后 - 训练集: {train_processed.shape}, 测试集: {test_processed.shape}")
        
        # ========== 3. 缺失值处理 ==========
        print("\n=== 缺失值处理 ===")
        train_filled = smart_missing_value_imputation(train_processed)
        test_filled = smart_missing_value_imputation(test_processed)
        
        # 分离特征和标签
        X_train = train_filled.drop('Transported', axis=1)
        y_train = train_filled['Transported']
        X_test = test_filled
        
        # ========== 4. 特征选择 ==========
        print("\n=== 特征选择 ===")
        selected_features = optimize_feature_selection(X_train, y_train, top_k=25)
        
        # 分离数值和分类特征
        numeric_features = [f for f in selected_features if f in X_train.select_dtypes(include=[np.number]).columns]
        categorical_features = [f for f in selected_features if f in X_train.select_dtypes(include=['object']).columns]
        
        print(f"选择特征 - 数值: {len(numeric_features)}, 分类: {len(categorical_features)}")
        
        # ========== 5. 模型训练 ==========
        print("\n=== 模型训练 ===")
        lgb1_pipeline = create_lgb1_model(numeric_features, categorical_features)
        
        # 交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("训练 LGB1...")
        cv_scores = cross_val_score(lgb1_pipeline, X_train[selected_features], y_train, 
                                  cv=skf, scoring='accuracy', n_jobs=1)
        mean_score = cv_scores.mean()
        print(f"LGB1 - 准确率: {mean_score:.5f} ± {cv_scores.std():.5f}")
        
        # 选择最佳模型（根据交叉验证分析，选择LGB1）
        best_score = mean_score
        best_model_name = "lgb1"
        best_model = lgb1_pipeline
        
        print(f"模型: {best_model_name}, 准确率: {best_score:.5f}")
        
        # ========== 6. 验证集评估 ==========
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
        
        # ========== 7. 全量训练和预测 ==========
        print("\n=== 全量训练 ===")
        best_model.fit(X_train[selected_features], y_train)
        y_test_pred = best_model.predict(X_test[selected_features])
        
        # ========== 8. 生成提交文件 ==========
        submission_filename = "spaceship_titanic_submission.csv"
        submission = pd.DataFrame({
            "PassengerId": test_ids,
            "Transported": y_test_pred.astype(bool)
        })
        submission.to_csv(submission_filename, index=False)
        
        # ========== 9. 结果汇总 ==========
        print("\n" + "="*50)
        print("训练和预测完成!")
        print("="*50)
        print(f"提交文件: {submission_filename}")
        print(f"使用模型: {best_model_name}")
        print(f"交叉验证准确率: {best_score:.5f}")
        print(f"验证集准确率: {val_accuracy:.5f}")
        
        # 最终准确率估计
        final_accuracy = max(best_score, val_accuracy)
        print(f"最终预期准确率: {final_accuracy:.5f}")
        
        # 预测分布统计
        transported_count = submission['Transported'].sum()
        total_count = len(submission)
        transport_rate = transported_count / total_count * 100
        print(f"\n预测分布:")
        print(f"  Transported=True: {transported_count}/{total_count} ({transport_rate:.1f}%)")
        
        return submission, final_accuracy
        
    except Exception as e:
        # 异常处理
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


if __name__ == "__main__":
    # 程序入口
    submission, final_accuracy = main()