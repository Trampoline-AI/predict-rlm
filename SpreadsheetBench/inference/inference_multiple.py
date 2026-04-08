import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

from llm_api import get_llm_response
from code_exec import get_exec_client, extract_code, exec_code
from prompt_format import PROMPT_FORMAT_SINGLE, PROMPT_DF_RCT_FORMAT , PROMPT_NO_DF_RCT_FORMAT


def gen_file_content(input_file):
    excel_file = pd.ExcelFile(input_file)
    sheet_names = excel_file.sheet_names
    excel_data = {}

    for sheet_name in sheet_names:
        df = excel_file.parse(sheet_name)
        len = opt.row if df.shape[0] > opt.row else df.shape[0]
        excel_data[sheet_name] = df.head(len).to_string()

    final_str = ""
    for sheet_name, sheet_str in excel_data.items():
        final_str += f"Sheet Name: {sheet_name}\n"
        final_str += sheet_str + "\n"
        final_str += "-" * 50 + "\n"
    
    return final_str


def gen_solution(opt):
    dataset_path = os.path.abspath(f'../data/{opt.dataset}')
    with open(f'{dataset_path}/dataset.json', 'r') as fp:
        dataset = json.load(fp)
    
    # check if output file folder exists
    output_file_path = f'{dataset_path}/outputs'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
        os.chmod(output_file_path, 0o777)

    # check if output file folder of the model exists
    output_file_path = f'{output_file_path}/single_{opt.model}'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
        os.chmod(output_file_path, 0o777)

    # create code execution client
    client = get_exec_client(opt.code_exec_url, opt.conv_id)
        
    for data in tqdm(dataset):
        file_name = f"1_{data['spreadsheet_path'].lstrip('spreadsheet/')}_input.xlsx"
        input_path = f"/mnt/data/{data['spreadsheet_path']}/{file_name}"
        output_path = f"/mnt/data/outputs/multi_{opt.setting}_{opt.model}/{file_name.rstrip(f'_input.xlsx')}_output.xlsx"
        find_input_path = f"{dataset_path}/{data['spreadsheet_path']}/{file_name}"

        # three setting: row_exec, react_exec, row_react_exec
        if opt.setting == 'row_exec':
            file_content = gen_file_content(find_input_path)
            prompt = PROMPT_FORMAT_SINGLE.format_map({
                'instruction': data['instruction'],
                'spreadsheet_path': input_path,
                'spreadsheet_content' : file_content,
                'instruction_type': data['instruction_type'],
                'answer_position': data['answer_position'],
                'max_turn_num' : opt.max_turn_num,
                'output_path': output_path
            })
        elif opt.setting == 'react_exec':
            prompt = PROMPT_NO_DF_RCT_FORMAT.format_map({
                'instruction': data['instruction'],
                'spreadsheet_path': input_path,
                'instruction_type': data['instruction_type'],
                'answer_position': data['answer_position'],
                'max_turn_num' : opt.max_turn_num,
                'output_path': output_path
            })
        elif opt.setting == 'row_react_exec':
            file_content = gen_file_content(find_input_path)
            prompt = PROMPT_DF_RCT_FORMAT.format_map({
                'instruction': data['instruction'],
                'spreadsheet_path': input_path,
                'spreadsheet_content' : file_content,
                'instruction_type': data['instruction_type'],
                'answer_position': data['answer_position'],
                'max_turn_num' : opt.max_turn_num,
                'output_path': output_path
            })
        else:
            print('Wrong multi-round setting.')
            exit(0)

        messages = [prompt]
        for _ in tqdm(range(opt.max_turn_num)):
            response = get_llm_response(messages, opt)
            messages.append(response)
            try:
                exec_result = exec_code(client, extract_code(response))
            except Exception as e:
                exec_result = 'Error occur when running code.'
            messages.append(exec_result)
            if os.path.exists(output_path.replace('/mnt/data', dataset_path)):
                break
        conv_result = {
            'id': data['id'],
            'instruction_type': data['instruction_type'],
            'conversation': messages,
            'solution': extract_code(response)
        }
        with open(f'outputs/conv_multi_{opt.setting}_{opt.model}.jsonl', 'a+') as fp:
            fp.write(json.dumps(conv_result, ensure_ascii=False) + '\n')


def run_solution(opt):
    client = get_exec_client(opt.code_exec_url, opt.conv_id)
    dataset_path = os.path.abspath(f'../data/{opt.dataset}')
    with open(f'{dataset_path}/outputs/conv_multi_{opt.setting}_{opt.model}.jsonl', 'r') as fp:
        conv_records = [json.loads(line) for line in fp.readlines()]
    for conv in tqdm(conv_records):
        for idx in range(2, 4):
            input_file = f"{idx}_{conv['id']}_input.xlsx"
            output_file = f"{idx}_{conv['id']}_output.xlsx"
            solution = conv['solution'].replace(f"1_{conv['id']}_input.xlsx", input_file)
            solution = solution.replace(f"1_{conv['id']}_output.xlsx", output_file)
            exec_result = exec_code(client, solution)


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generation.")

    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--api_key', type=str, default="", help='the api key of model')
    parser.add_argument('--base_url', type=str, default="", help='the base url of model')
    parser.add_argument('--setting', type=str, help='three setting: row_exec, react_exec, row_react_exec')
    parser.add_argument('--dataset', type=str, default="sample_data_200", help='dataset name')
    parser.add_argument('--code_exec_url', type=str, default="http://localhost:8081/execute", help='code execution docker url')
    parser.add_argument('--conv_id', type=str, default="EVAL", help='code execution conversation id')
    parser.add_argument('--max_turn_num', type=int, default=5, help='max turn number of conversation')
    parser.add_argument('--row', type=int, default=5, help='the number of rows provided in the prompt')
    
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_option()
    print(opt)

    gen_solution(opt)
    run_solution(opt)
