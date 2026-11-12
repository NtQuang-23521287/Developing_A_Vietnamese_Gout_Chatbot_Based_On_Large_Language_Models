import json
import os

import pandas as pd


def main():
    output_dir = "data/raw/testset"
    os.makedirs(output_dir, exist_ok=True)
    
    target_filename = "Group3_Test_Queries.csv"
    input_file = None
    
    print(f"Đang quét tìm {target_filename}...")
    # Tự động quét toàn bộ thư mục từ vị trí hiện tại
    for root, _dirs, files in os.walk("."):
        if target_filename in files:
            input_file = os.path.join(root, target_filename)
            break
            
    if not input_file:
        print(f"LỖI: Không tìm thấy {target_filename} trong dự án.")
        return
        
    print(f"Đã tìm thấy dữ liệu tại: {input_file}")
    df = pd.read_csv(input_file)
    output_file = f"{output_dir}/Group3_Testset_cases_v02.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            case = {
                "case_id": f"group3_{idx+1:03d}",
                "source_row_id": f"row_{idx}",
                "scenario": str(row.get('target_disease', 'Bệnh Gút')),
                "subset": "group3_candidate_v02",
                "turns": [
                    {
                        "turn_id": "turn_1",
                        "user": str(row['test_query']).strip()
                    }
                ]
            }
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
            
    print(f"Đã chuyển đổi thành công {len(df)} câu hỏi.")
    print(f"File JSONL đã sẵn sàng tại: {output_file}")

if __name__ == "__main__":
    main()