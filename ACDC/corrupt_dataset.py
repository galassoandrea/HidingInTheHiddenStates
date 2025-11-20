import os
import random
import re

import pandas as pd

def corrupt_sentence(sentence, topic):
    sentence = sentence.rstrip(".")
    words = sentence.split()
    if topic == "animals":
        if re.search(r'\bhuman uses for\b', sentence.lower()):
            idx = next(i for i, word in enumerate(words) if word == "for")
            clean_animal = words[idx + 1]
            other_animals = [a for a in animals if a != clean_animal]
            corr_animal = random.choice(other_animals)
            words[idx + 1] = corr_animal
        elif re.search(r'\bhas a habitat of\b', sentence.lower()):
            idx = next(i for i, word in enumerate(words) if word == "habitat")
            clean_habitat = words[idx + 2]
            other_habitats = [a for a in habitats if a != clean_habitat and clean_habitat not in a]
            corr_habitat = random.choice(other_habitats)
            words[idx + 2] = corr_habitat
        elif re.search(r'\bfor locomotion\b', sentence.lower()):
            idx = next(i for i, word in enumerate(words) if word == "for")
            clean_movement = words[idx - 1]
            other_movements = [m for m in movements if m != clean_movement]
            corr_movement = random.choice(other_movements)
            words[idx - 1] = corr_movement
        elif re.search(r'\bhas a diet of\b', sentence.lower()):
            idx = next(i for i, word in enumerate(words) if word == "diet")
            clean_diet = words[idx + 2]
            other_diets = [d for d in diets if d != clean_diet]
            corr_diet = random.choice(other_diets)
            words[idx + 2] = corr_diet
        elif re.search(r'\b(?:is a|is an)\b', sentence.lower()):
            idx = next(i for i, word in enumerate(words) if (word == "a" or word == "an") and words[i - 1] == "is")
            clean_species = words[idx + 1]
            other_species = [a for a in species if a != clean_species]
            corr_species = random.choice(other_species)
            words[idx + 1] = corr_species
        else:
            clean_animal = words[1]
            other_animals = [a for a in animals if a != clean_animal]
            corr_animal = random.choice(other_animals)
            words[1] = corr_animal
    elif topic == "elements":
        if "appears" in words:
            state = words[-1]
            other_states = [a for a in states if a != state]
            corr_state = random.choice(other_states)
            words[-1] = corr_state
        else:
            clean_element = words[0]
            other_elements = [a for a in elements if a != clean_element]
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
            other_inventors = [a for a in inventors if a != clean_inventor]
            corr_inventor = random.choice(other_inventors)
            words[:idx] = corr_inventor.split()
        elif "lived" in words:
            idx = words.index("lived")
            clean_country = " ".join(words[idx + 2:])
            other_countries = [a for a in countries if a != clean_country]
            corr_country = random.choice(other_countries)
            words[idx + 2:] = corr_country.split()
            words = words[:idx + 3]
    elif topic == "companies":
        if re.search(r'\bhas headquarters\b', sentence.lower()):
            idx = words.index("headquarters")
            clean_country = " ".join(words[idx + 2:])
            other_countries = [a for a in countries if a != clean_country]
            corr_country = random.choice(other_countries)
            words[idx + 2:] = corr_country.split()
            words = words[:idx + 3]
        elif any(w in words for w in ["engages", "operates", "provides", "produces", "manages"]):
            idx = next(i for i, word in enumerate(words) if word in ["engages", "operates", "provides", "produces", "manages"])
            clean_company = " ".join(words[:idx])
            other_companies = [a for a in companies if a != clean_company]
            corr_company = random.choice(other_companies)
            words[:idx] = corr_company.split()
        else:
            idx = next(i for i, word in enumerate(words) if word == "is")
            clean_company = " ".join(words[:idx])
            other_companies = [a for a in companies if a != clean_company]
            corr_company = random.choice(other_companies)
            words[:idx] = corr_company.split()
    # Join words back into a sentence and add final period
    sentence = " ".join(words) + "."
    return sentence

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

list_of_datasets = ["animals", "cities", "elements", "companies", "inventions"]

# Define pools of elements for corruption
animals = ['beaver', 'leopard', 'swan', 'polar bear', 'wolverine', 'salmon', 'rhinoceros', 'manta', 'gecko',
                'giant anteater', 'snake', 'skunk', 'hippopotamus', 'cow', 'vulture', 'deer', 'sparrow', 'seagull',
                'mongoose', 'rat', 'crocodile', 'flamingo', 'tapir', 'jellyfish', 'walrus', 'hedgehog', 'hamster',
                'giraffe', 'ostrich', 'dog', 'slug', 'tortoise', 'hummingbird', 'tiger', 'camel', 'zebra', 'lobster',
                'kangaroo', 'aardvark', 'dolphin', 'manta ray', 'tuna', 'elephant', 'peacock', 'goldfish', 'raccoon',
                'alpaca', 'axolotl', 'armadillo']
movements = ['walking', 'running', 'swimming', 'jumping', 'flying']
habitats = ['forest/grassland', 'marine/polar', 'coastal/alkaline lakes', 'freshwater', 'savanna', 'desert',
                 'forest/urban', 'farmland', 'arctic/subarctic', 'mountain']
diets = ['insectivore', 'carnivore', 'herbivore', 'omnivore', 'nectar']
species = ['mammal', 'bird', 'fish', 'reptile', 'amphibian', 'insect', 'arachnid', 'crustacean', 'mollusk',
                'cnidarian']
elements = ['Tantalum', 'Calcium', 'Gadolinium', 'Samarium', 'Cerium', 'Iridium', 'Rhenium', 'Scandium', 'Nickel',
                 'Thallium', 'Silver', 'Oxygen', 'Actinium', 'Promethium', 'Astatine', 'Osmium', 'Platinum', 'Tin',
                 'Nitrogen', 'Beryllium', 'Arsenic', 'Lead', 'Mercury', 'Fluorine', 'Lanthanum', 'Radium', 'Iron',
                 'Zirconium', 'Praseodymium', 'Lithium', 'Francium', 'Ruthenium', 'Iodine', 'Neon', 'Copper', 'Erbium',
                 'Krypton', 'Rubidium', 'Thorium', 'Protactinium', 'Rhodium', 'Antimony', 'Boron', 'Bismuth',
                 'Tellurium', 'Titanium', 'Cadmium', 'Sulfur', 'Holmium', 'Gallium', 'Technetium', 'Tungsten']
companies = ['China Life Insurance', 'JPMorgan Chase', 'Bank of America', 'Mizuho Financial', 'UBS Group',
                  'American Express', 'Airbus Group', 'General Motors', 'Abbott Laboratories', 'Berkshire Hathaway',
                  'Johnson & Johnson', 'Bristol Myers Squibb', 'Anglo American', 'Tencent Holdings',
                  'Bank of Nova Scotia', 'Procter & Gamble', 'Bayerische Motoren Werke (BMW)', 'Rio Tinto',
                  'Intesa Sanpaolo', 'HCA Healthcare', 'IBM', 'Munich Reinsurance', 'ArcelorMittal',
                  'China Merchants Bank', 'Alibaba Group', 'Truist Financial',
                  'Industrial and Commercial Bank of China', 'The Home Depot', 'Goldman Sachs Group', 'America Movil',
                  'Medtronic', 'AXA', 'Nippon Telegraph & Telephone', 'Stellantis', 'Iberdrola']
inventors = ['Charles Wheatstone', 'Alfred Nobel', 'Edwin Herbert Hall', 'Emile Berliner', 'Frederick Walton',
                  'William Congreve', 'Fridtjof Nansen', 'Gustaf Dalén', 'Evangelista Torricelli', 'George Pullman',
                  'George Devol', 'Henri Giffard', 'Marvin Camras', 'Carlos Glidden', 'Charles Francis Richter',
                  'Alexander Graham Bell', 'William Sturgeon', 'Ernesto Blanco', 'Clarence Birdseye', 'Gideon Sundback',
                  'John Shepherd-Barron', 'Pierre Curie', 'Wilhelm Conrad Röntgen', 'Benjamin Franklin',
                  'John Wesley Hyatt', 'Louis Pasteur', 'Sir Frank Whittle', 'James Watt', 'Edwin Link', 'Maria Telkes',
                  'Igor Tamm', 'Chester Carlson', 'Josephine Cochrane', 'Vint Cerf', 'Ralph H. Baer', 'Whitcomb Judson',
                  'Joseph Glidden', 'Henry Ford', 'Biruté Galdikas', 'John Callcott Horsley', 'Richard Trevithick',
                  'Giovanni Caselli', 'Jack Kilby', 'Charles Goodyear', 'Lloyd Groff Copeman', 'Valdemar Poulsen',
                  'Herbert Akroyd Stuart', 'Rudolf Diesel']
countries = ['Russia', 'Canada', 'France', 'Germany', 'Italy', 'Spain', 'Australia', 'Brazil', 'India', 'China',
                  'Japan', 'Mexico', 'South Africa', 'Egypt', 'Turkey', 'Argentina', 'Colombia', 'Indonesia']
states = ['Solid', 'Liquid', 'Gas']


for topic in list_of_datasets:
    # Load the dataframe
    df_path = os.path.join(root_dir, "resources", f"{topic}_true_false.csv")
    df = pd.read_csv(df_path)
    # Extract "true" sentences
    df = df[df["label"] == 1].iloc[:200]
    # Extract clean sentences and labels
    clean_statements = df['statement'].tolist()
    labels = df['label'].tolist()
    # Generate corrupted sentences
    corrupted_statements = [corrupt_sentence(sentence, topic) for sentence in clean_statements]
    # Save corrupted sentences to new CSV
    corrupted_df = pd.DataFrame({'clean_statement': clean_statements, 'corrupted_statement': corrupted_statements, 'label': labels})
    corrupted_df_path = os.path.join(root_dir, "resources", f"{topic}_clean_corrupted.csv")
    corrupted_df.to_csv(corrupted_df_path, index=False)