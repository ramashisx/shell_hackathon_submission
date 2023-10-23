import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import argparse
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


def main(args):

    data = pd.read_json("../data/datafinal.json")

    mlb = MultiLabelBinarizer()
    device = "cuda"
    mlb.fit(data["CodeList"])

    test_data = data[~data["ContainsCode"].apply(lambda x: isinstance(x, bool))].copy()
    submission_file_path = f"./{args.output}.csv"
    raw_file_path = f"./{args.output}_raw.csv"

        
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    for t in mlb.classes_:
        if t != tokenizer.decode(tokenizer(t)["input_ids"], skip_special_tokens=True):
            print("tokenizer not supported")
            exit()


    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    model = model.to(device)
    model.generation_config.max_length = 512

    i = 49
    text = test_data["Text"].iloc[i]
    # text = """Software development has changed significantly over the years. Nowadays, it's common to use high-level programming languages like JavaScript, which are more developer-friendly. A simple JavaScript code to display a message in the console would be:\n\nconsole.log('Hello, World!');\n\nDespite their simplicity, high-level languages are powerful tools for creating complex applications."""
    answer = test_data["CodeList"].iloc[i]

    print("===================")
    print("One Sample")
    print(text)
    print("===========")
    print(answer)

    output = model.generate(**tokenizer.encode_plus(text, return_tensors="pt").to(device), max_length=256)
    print(tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace('\\n', '\n').replace('\\n', '\n'))

    print("One Sample End")
    print("---------------------------------")


    code_list = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        if isinstance(row["ContainsCode"], str):
            output = model.generate(**tokenizer.encode_plus(row["Text"], return_tensors="pt").to(device), max_length=512)
            result = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            code_list.append(result.replace('\n', '\\n').replace('\t', '\\t'))
        else:
            code_list.append(row["CodeList"])



    print("-----------\nActual MLB Clasees")
    mlb = MultiLabelBinarizer()
    s1 = data["CodeList"]
    t = mlb.fit_transform(s1)
    print(mlb.classes_)
    print("-------------------------")

    print("-----------\nPrdicted MLB Clasees")
    t1 = mlb.fit_transform(code_list)
    print(mlb.classes_)

    submission = pd.DataFrame(t1)
    submission.to_csv(submission_file_path, index=False)

    data["CodeList"] = code_list
    data.to_csv(raw_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission"
    )
    args = parser.parse_args()
    main(args)