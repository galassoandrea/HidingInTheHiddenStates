import pandas as pd

def build_prompt_dataset_original(list_of_datasets, n_shots):
    for dataset_to_use in list_of_datasets:
        df = pd.read_csv(f"resources/{dataset_to_use}_true_false.csv").head(500)
        rows = []
        for i in range(n_shots, len(df)):
            prompt = ""
            for j in range(n_shots - 1, 0, -1):
                sentence = df.at[i - j, 'statement']
                sentence = sentence.rstrip(". ")
                sentence += ": "
                truth = df.at[i - j, 'label']
                sentence += "true. " if truth == 1 else "false. "
                prompt += sentence
            prompt += df.at[i, 'statement']
            prompt = prompt.rstrip(". ")
            prompt += ": "

            # Add row to list
            rows.append({
                "prompt": prompt,
                "label": df.at[i, "label"]
            })

        # Create dataframe
        prompts_df = pd.DataFrame(rows)
        prompts_df.to_csv(f"resources/{dataset_to_use}_few_shot_prompt.csv", index=False)


def build_prompt_dataset_clean_corrupted(list_of_datasets, n_shots):
    for dataset_to_use in list_of_datasets:
        df = pd.read_csv(f"resources/{dataset_to_use}_clean_corrupted.csv")
        rows = []
        for i in range(n_shots, len(df)):
            clean_prompt = ""
            corrupted_prompt = ""
            for j in range(n_shots - 1, 0, -1):
                sentence = df.at[i - j, 'clean_statement']
                sentence = sentence.rstrip(". ")
                sentence += ": "
                truth = df.at[i - j, 'label']
                sentence += "true. " if truth == 1 else "false. "
                clean_prompt += sentence
                corrupted_prompt += sentence
            clean_prompt += df.at[i, 'clean_statement']
            clean_prompt = clean_prompt.rstrip(". ")
            clean_prompt += ": "
            corrupted_prompt += df.at[i, 'corrupted_statement']
            corrupted_prompt = corrupted_prompt.rstrip(". ")
            corrupted_prompt += ": "

            # Add row to list
            rows.append({
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                "label": 1 - df.at[i, "label"]
            })

        # Create dataframe
        prompts_df = pd.DataFrame(rows)
        prompts_df.to_csv(f"resources/{dataset_to_use}_clean_corrupted_prompt.csv", index=False)

list_of_datasets = [
    "animals",
    "cities",
    "elements",
    "companies",
    "inventions",
    "facts"
]

n_shots = 3

# Build few-shot prompt dataset from original sentences
build_prompt_dataset_original(list_of_datasets=["animals", "cities", "elements", "companies", "inventions", "facts"],
                              n_shots=n_shots)

# Build few-shot prompt dataset with clean and corrupted sentences for circuit discovery
build_prompt_dataset_clean_corrupted(list_of_datasets=["animals", "cities", "elements", "companies", "inventions"],
                              n_shots=n_shots)