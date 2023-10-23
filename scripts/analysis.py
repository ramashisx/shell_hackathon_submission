import argparse
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM



def main(args):
    data = pd.read_json("../data/datafinal.json")

    mlb = MultiLabelBinarizer()
    device = "cuda"
    mlb.fit(data["CodeList"])

    train_data = data[data["ContainsCode"].apply(lambda x: isinstance(x, bool))].copy()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    for t in mlb.classes_:
        if t != tokenizer.decode(tokenizer(t)["input_ids"], skip_special_tokens=True):
            print("tokenizer not supported")
            exit()
    
    count = 0
    y_test = []
    y_pred = []
    for _, row in tqdm(train_data.iterrows(), total=len(train_data)):
        output = model.generate(**tokenizer.encode_plus(row["Text"], return_tensors="pt").to(device), max_length=256)
        output = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        x1 = mlb.transform([row["CodeList"].replace('\n', '\\n').replace('\t', '\\t')])
        x2 = mlb.transform([output.replace('\n', '\\n').replace('\t', '\\t')])
        y_test.append(row["CodeList"].replace('\n', '\\n').replace('\t', '\\t'))
        y_pred.append(output.replace('\n', '\\n').replace('\t', '\\t'))
        if abs(x1-x2).sum() != 0:
            print(row["Text"])
            print("===============")
            print(row["CodeList"])
            print("++++++++++++")
            print(output)
            print("----------------"*100)
            count += 1

    print(accuracy_score(mlb.transform(y_test), mlb.transform(y_pred)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        required=True
    )
    args = parser.parse_args()
    main(args)
