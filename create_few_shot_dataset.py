import pandas as pd

def build_prompt_dataset_original(list_of_datasets, n_shots):
    for dataset_to_use in list_of_datasets:
        df = pd.read_csv(f"resources/{dataset_to_use}_true_false.csv").head(500)
        rows = []
        for i in range(n_shots, len(df)):
            prompt = ""
            for j in range(n_shots - 1, 0, -1):
                sentence = df.at[i - j, 'statement'].rstrip(". ")
                truth = df.at[i - j, 'label']
                label = "true." if truth == 1 else "false."
                prompt += f"{sentence}: {label}\n"  # <-- newline instead of space

            # Final example (the one to be completed by the model)
            current = df.at[i, 'statement'].rstrip(". ")
            prompt += f"{current}: "

            # Add row to list
            rows.append({
                "prompt": prompt,
                "label": df.at[i, "label"]
            })

        # Create dataframe
        prompts_df = pd.DataFrame(rows)
        prompts_df.to_csv(f"resources/{dataset_to_use}_few_shot_prompt.csv", index=False)