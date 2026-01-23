import pandas as pd
import glob
import os
import numpy as np

# ================= 配置项 =================
# CSV 文件所在的文件夹路径 (如果脚本就在同级目录，用 '.' 即可)
DATA_DIR = './processed' 
# 想要生成的文件名
OUTPUT_FILE = './data/G28_RawPrice.txt'
# =========================================

def main():
    # 1. 寻找文件夹里所有的 csv 文件
    # 假设您的文件名格式类似 "AUDNZD_Processed_1H.csv"
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_Processed_1H.csv"))
    
    if len(csv_files) == 0:
        print("❌ 错误：在当前目录下没有找到文件名包含 '_Processed_1H.csv' 的文件！")
        return

    print(f"🔍 找到了 {len(csv_files)} 个数据文件，准备开始合并...")
    
    # 用于存放所有货币对数据的列表
    series_list = []
    
    for file_path in sorted(csv_files):
        # 提取货币对名称，例如从 "AUDNZD_Processed_1H.csv" 中提取 "AUDNZD"
        filename = os.path.basename(file_path)
        pair_name = filename.split('_')[0] 
        print(f"   -> 正在读取: {pair_name} ...")
        
        try:
            # 读取 CSV
            df = pd.read_csv(file_path)
            
            # 确保 'time' 列是时间格式，并设为索引
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # 只取 'Close' 列，并重命名为货币对名称
            close_series = df[['Close']].rename(columns={'Close': pair_name})
            
            # 去除重复的时间索引 (以防万一)
            close_series = close_series[~close_series.index.duplicated(keep='first')]
            
            series_list.append(close_series)
            
        except Exception as e:
            print(f"⚠️ 读取 {filename} 时出错: {e}")

    # 2. 合并数据 (Merge)
    # 使用 outer join 确保并集，保证时间轴是完整的
    print("⏳ 正在按时间轴对齐合并...")
    final_df = pd.concat(series_list, axis=1)
    
    # 按时间排序
    final_df.sort_index(inplace=True)
    
    print(f"   原始数据形状: {final_df.shape} (行=时间, 列=货币对)")
    
    # 3. 处理缺失值 (Data Cleaning)
    # 外汇数据因为周末休市，大家应该都是空的，可以drop
    # 或者某些时刻个别货币缺失，用 ffill (前向填充)
    
    # 策略：先用前向填充(fill forward)补全偶尔的交易缺失
    final_df.fillna(method='ffill', inplace=True)
    # 再用后向填充(back fill)补全开头可能的缺失
    final_df.fillna(method='bfill', inplace=True)
    
    # 如果还有整行都是空的(比如周末)，直接丢弃
    original_len = len(final_df)
    final_df.dropna(axis=0, how='any', inplace=True)
    print(f"   去除包含 NaN 的行后形状: {final_df.shape} (删除了 {original_len - len(final_df)} 行)")

    # 4. 保存为 txt 文件
    # header=False (不保存列名), index=False (不保存时间列)
    # MTGNN 只要纯数字矩阵
    final_df.to_csv(OUTPUT_FILE, sep=',', header=False, index=False)
    
    print("="*30)
    print(f"✅ 处理完成！")
    print(f"📂 输出文件已保存为: {OUTPUT_FILE}")
    print(f"📊 最终矩阵大小: {final_df.shape}")
    print("   (行数应当作为 seq_in_len 的参考，列数应为 28)")
    print("="*30)
    
    # 简单检查一下生成的数据
    # print(final_df.head())

if __name__ == "__main__":
    main()