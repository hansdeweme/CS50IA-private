import csv
import itertools
import sys

PROBS = {
    "gene": {2: 0.01, 1: 0.03, 0: 0.96},
    "trait": {
        2: {True: 0.65, False: 0.35},
        1: {True: 0.56, False: 0.44},
        0: {True: 0.01, False: 0.99},
    },
    "mutation": 0.01,
}


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py familyX.csv, where X = 0, 1 or 2")
    people = load_data(sys.argv[1])

    probabilities = {
        person: {
            "gene": {2: 0, 1: 0, 0: 0},
            "trait": {True: 0, False: 0},
        }
        for person in people
    }

    names = set(people)
    for have_trait in powerset(names):
        if any(
            people[person]["trait"] is not None
            and people[person]["trait"] != (person in have_trait)
            for person in names
        ):
            continue

        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    normalize(probabilities)

    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}: ")
            for value in probabilities[person][field]:
                print(f"    {value}: {probabilities[person][field][value]:.4f}")


def load_data(filename):
    data = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else False if row["trait"] == "0" else None),
            }
    return data


def powerset(s):
    return [
        set(s)
        for s in itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))
    ]


def inherit_prob(parent, one_gene, two_genes):
    parent_genes = 2 if parent in two_genes else 1 if parent in one_gene else 0
    return {0: PROBS["mutation"], 1: 0.5, 2: 1 - PROBS["mutation"]}[parent_genes]


def joint_probability(people, one_gene, two_genes, have_trait):
    joint_prob = 1
    
    for person in people:
        person_genes = 2 if person in two_genes else 1 if person in one_gene else 0
        person_trait = person in have_trait
        mother, father = people[person]["mother"], people[person]["father"]
        
        if not mother and not father:
            person_prob = PROBS["gene"][person_genes]
        else:
            mother_prob = inherit_prob(mother, one_gene, two_genes)
            father_prob = inherit_prob(father, one_gene, two_genes)
            
            if person_genes == 2:
                person_prob = mother_prob * father_prob
            elif person_genes == 1:
                person_prob = (1 - mother_prob) * father_prob + (1 - father_prob) * mother_prob
            else:
                person_prob = (1 - mother_prob) * (1 - father_prob)
        
        trait_prob = PROBS["trait"][person_genes][person_trait]
        joint_prob *= person_prob * trait_prob
    
    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    for person in probabilities:
        person_genes = 2 if person in two_genes else 1 if person in one_gene else 0
        person_trait = person in have_trait
        probabilities[person]["gene"][person_genes] += p
        probabilities[person]["trait"][person_trait] += p


def normalize(probabilities):
    for person in probabilities:
        for field in ["gene", "trait"]:
            total = sum(probabilities[person][field].values())
            probabilities[person][field] = {k: v / total for k, 
                                            v in probabilities[person][field].items()}


if __name__ == "__main__":
    main()
