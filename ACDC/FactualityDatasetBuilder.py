import os
from dataclasses import dataclass
import random
from typing import List
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def pad_sequences(tokens: List[torch.Tensor], max_length) -> torch.Tensor:
    """Pad each token sequence to the maximum length in the batch"""
    padded_tokens = [
        F.pad(t, (0, max_length - t.shape[0]))
        for t in tokens
    ]
    return torch.stack(padded_tokens, dim=0)


class FactualityDataset(Dataset):
    def __init__(self, examples):
        """
        Args:
            examples: List of FactualityExample objects with pre-tokenized data
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'clean_tokens': torch.tensor(example.clean_tokens, dtype=torch.long),
            'corrupted_tokens': torch.tensor(example.corrupted_tokens, dtype=torch.long),
            'label': torch.tensor(example.label, dtype=torch.long)
        }


@dataclass
class FactualityExample:
    """Represents a single Factuality example"""
    clean_tokens: List[str]
    corrupted_tokens: List[str]
    label: int


class FactualityDatasetBuilder:
    """Builds a dataset for the Factuality task"""

    def __init__(self, model):
        # Load each dataframe and create a combined dataframe with a topic column
        self.df = pd.DataFrame()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        for topic in ["animals", "cities", "elements", "companies", "inventions"]:
            df_path = os.path.join(root_dir, "resources", f"{topic}_true_false.csv")
            df = pd.read_csv(df_path, nrows=10)
            df["topic"] = topic
            self.df = pd.concat([self.df, df], ignore_index=True)
        self.model = model
        self.system_role = ("Is the following sentence true or false? Answer with 1 if the sentence is true and"
                            " 0 if the sentence is false: ")

        # Pools of elements for corruption
        self.animals = ['beaver', 'leopard', 'swan', 'polar bear', 'wolverine', 'salmon', 'rhinoceros', 'manta', 'gecko', 'giant anteater', 'snake', 'skunk', 'hippopotamus', 'cow', 'vulture', 'deer', 'sparrow', 'seagull', 'mongoose', 'rat', 'crocodile', 'flamingo', 'tapir', 'jellyfish', 'walrus', 'hedgehog', 'hamster', 'giraffe', 'ostrich', 'dog', 'slug', 'tortoise', 'hummingbird', 'tiger', 'camel', 'zebra', 'lobster', 'kangaroo', 'aardvark', 'dolphin', 'manta ray', 'tuna', 'elephant', 'peacock', 'goldfish', 'raccoon', 'alpaca', 'axolotl', 'armadillo']
        self.habitats = ['forest/grassland', 'marine/polar', 'coastal/alkaline lakes', 'freshwater', 'savanna', 'desert', 'forest/urban', 'farmland', 'arctic/subarctic', 'mountain']
        self.species = ['mammal', 'bird', 'fish', 'reptile', 'amphibian', 'insect', 'arachnid', 'crustacean', 'mollusk', 'cnidarian']
        self.elements = ['Tantalum', 'Calcium', 'Gadolinium', 'Samarium', 'Cerium', 'Iridium', 'Rhenium', 'Scandium', 'Nickel', 'Thallium', 'Silver', 'Oxygen', 'Actinium', 'Promethium', 'Astatine', 'Osmium', 'Platinum', 'Tin', 'Nitrogen', 'Beryllium', 'Arsenic', 'Lead', 'Mercury', 'Fluorine', 'Lanthanum', 'Radium', 'Iron', 'Zirconium', 'Praseodymium', 'Lithium', 'Francium', 'Ruthenium', 'Iodine', 'Neon', 'Copper', 'Erbium', 'Krypton', 'Rubidium', 'Thorium', 'Protactinium', 'Rhodium', 'Antimony', 'Boron', 'Bismuth', 'Tellurium', 'Titanium', 'Cadmium', 'Sulfur', 'Holmium', 'Gallium', 'Technetium', 'Tungsten']
        self.companies = ['China Life Insurance','JPMorgan Chase','Bank of America','Mizuho Financial','UBS Group','American Express','Airbus Group','General Motors','Abbott Laboratories','Berkshire Hathaway','Johnson & Johnson','Bristol Myers Squibb','Anglo American','Tencent Holdings','Bank of Nova Scotia','Procter & Gamble','Bayerische Motoren Werke (BMW)','Rio Tinto','Intesa Sanpaolo','HCA Healthcare','IBM','Munich Reinsurance','ArcelorMittal','China Merchants Bank','Alibaba Group','Truist Financial','Industrial and Commercial Bank of China','The Home Depot','Goldman Sachs Group','America Movil','Medtronic','AXA','Nippon Telegraph & Telephone','Stellantis','Iberdrola']
        self.inventors = ['Charles Wheatstone','Alfred Nobel','Edwin Herbert Hall','Emile Berliner','Frederick Walton','William Congreve','Fridtjof Nansen','Gustaf Dalén','Evangelista Torricelli','George Pullman','George Devol','Henri Giffard','Marvin Camras','Carlos Glidden','Charles Francis Richter','Alexander Graham Bell','William Sturgeon','Ernesto Blanco','Clarence Birdseye','Gideon Sundback','John Shepherd-Barron','Pierre Curie','Wilhelm Conrad Röntgen','Benjamin Franklin','John Wesley Hyatt','Louis Pasteur','Sir Frank Whittle','James Watt','Edwin Link','Maria Telkes','Igor Tamm','Chester Carlson','Josephine Cochrane','Vint Cerf','Ralph H. Baer','Whitcomb Judson','Joseph Glidden','Henry Ford','Biruté Galdikas','John Callcott Horsley','Richard Trevithick','Giovanni Caselli','Jack Kilby','Charles Goodyear','Lloyd Groff Copeman','Valdemar Poulsen','Herbert Akroyd Stuart','Rudolf Diesel']
        self.countries = ['Russia', 'Canada', 'France', 'Germany', 'Italy', 'Spain', 'Australia', 'Brazil', 'India', 'China', 'Japan', 'Mexico', 'South Africa', 'Egypt', 'Turkey', 'Argentina', 'Colombia', 'Indonesia']
        self.states = ['Solid', 'Liquid', 'Gas']

    def corrupt_sentence(self, sentence, topic):
        sentence = sentence.rstrip(".")
        words = sentence.split()
        if topic == "animals":
            if "human uses for" in sentence.lower():
                idx = next(i for i, word in enumerate(words) if word == "for")
                clean_animal = words[idx+1]
                other_animals = [a for a in self.animals if a != clean_animal]
                corr_animal = random.choice(other_animals)
                words[idx+1] = corr_animal
            elif "has a habitat of" in sentence.lower():
                idx = next(i for i, word in enumerate(words) if word == "habitat")
                clean_habitat = words[idx + 2]
                other_habitats = [a for a in self.habitats if a != clean_habitat and clean_habitat not in a]
                corr_habitat = random.choice(other_habitats)
                words[idx + 2] = corr_habitat
            elif "is a" in sentence:
                idx = next(i for i, word in enumerate(words) if word == "a")
                clean_species = words[idx + 1]
                other_species = [a for a in self.species if a != clean_species]
                corr_species = random.choice(other_species)
                words[idx + 1] = corr_species
            else:
                clean_animal = words[1]
                other_animals = [a for a in self.animals if a != clean_animal]
                corr_animal = random.choice(other_animals)
                words[1] = corr_animal
        elif topic == "elements":
            if "appears" in words:
                state = words[-1]
                other_states = [a for a in self.states if a != state]
                corr_state = random.choice(other_states)
                words[-1] = corr_state
            else:
                clean_element = words[0]
                other_elements = [a for a in self.elements if a != clean_element]
                corr_element = random.choice(other_elements)
                words[0] = corr_element
        elif topic == "cities":
            if "city" in words:
                idx = words.index("city")
                words[idx] = "country"
            elif "country" in words:
                idx = words.index("country")
                words[idx] = "city"
        elif topic == "inventions":
            if "invented" in words:
                idx = words.index("invented")
                clean_inventor = " ".join(words[:idx])
                other_inventors = [a for a in self.inventors if a != clean_inventor]
                corr_inventor = random.choice(other_inventors)
                words[:idx] = corr_inventor.split()
            elif "lived" in words:
                idx = words.index("lived")
                clean_country = " ".join(words[idx+2:])
                other_countries = [a for a in self.countries if a != clean_country]
                corr_country = random.choice(other_countries)
                words[idx+2:] = corr_country.split()
                words = words[:idx+3]
        elif topic == "companies":
            if "has headquarters" in sentence.lower():
                idx = words.index("headquarters")
                clean_country = " ".join(words[idx+2:])
                other_countries = [a for a in self.countries if a != clean_country]
                corr_country = random.choice(other_countries)
                words[idx+2:] = corr_country.split()
                words = words[:idx+3]
            elif "engages" in words or "operates" in words:
                idx = next(i for i, word in enumerate(words) if word in ["engages", "operates"])
                clean_company = " ".join(words[:idx])
                other_companies = [a for a in self.companies if a != clean_company]
                corr_company = random.choice(other_companies)
                words[:idx] = corr_company.split()
            else:
                idx = next(i for i, word in enumerate(words) if word == "is")
                clean_company = " ".join(words[:idx])
                other_companies = [a for a in self.companies if a != clean_company]
                corr_company = random.choice(other_companies)
                words[:idx] = corr_company.split()
        sentence = " ".join(words)
        return sentence

    def build_single_prompt(self, example):
        """Builds a single prompt for a given example"""
        clean_prompt = self.system_role + f'\nStatement: {example["statement"]}\nEvaluation: '
        corrupted_statement = self.corrupt_sentence(example["statement"], example["topic"])
        corrupted_prompt = self.system_role + f'\nStatement: {corrupted_statement}\nEvaluation: '
        clean_tokens = self.model.to_tokens(clean_prompt, prepend_bos=True).squeeze(0)
        corrupted_tokens = self.model.to_tokens(corrupted_prompt, prepend_bos=True).squeeze(0)
        return FactualityExample(
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            label=example['label']
        )

    def build_dataset(self):
        """Builds the complete dataset"""
        dataset = []
        clean_tokens = []
        corrupted_tokens = []
        for _, row in self.df.iterrows():
            example = {
                "statement": row['statement'],
                "label": row['label'],
                "topic": row['topic']
            }
            factuality_example = self.build_single_prompt(example)
            clean_tokens.append(factuality_example.clean_tokens)
            corrupted_tokens.append(factuality_example.corrupted_tokens)
            dataset.append(factuality_example)
        # Pad sequences to the maximum length
        max_length = max(token.shape[0] for token in clean_tokens)
        padded_clean_tokens = pad_sequences(clean_tokens, max_length)
        padded_corrupted_tokens = pad_sequences(corrupted_tokens, max_length)
        for i, example in enumerate(dataset):
            example.clean_tokens = padded_clean_tokens[i]
            example.corrupted_tokens = padded_corrupted_tokens[i]
        return dataset


