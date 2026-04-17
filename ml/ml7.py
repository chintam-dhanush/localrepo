

# !pip install pgmpy

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

heartDisease = pd.read_csv('/content/heartt.csv')

print("First 5 rows of dataset:")
print(heartDisease.head())

from pgmpy.models import DiscreteBayesianNetwork

model = DiscreteBayesianNetwork([
    ('age', 'heartdisease'),
    ('gender', 'heartdisease'),
    ('exang', 'heartdisease'),
    ('cp', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'chol')
])

print("\nBayesian Network Structure:")
print(model.edges())

model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

print("\nConditional Probability Tables (CPTs):")
for cpd in model.get_cpds():
    print(cpd)

print("\nInferencing with Bayesian Network:")

HeartDiseasetest_infer = VariableElimination(model)

# Query 1
print("\n1. Probability of Heart Disease given evidence = restecg")

q1 = HeartDiseasetest_infer.query(
    variables=['heartdisease'],
    evidence={'restecg': 1}
)

print(q1)

q2 = HeartDiseasetest_infer.query(
    variables=['heartdisease'],
    evidence={'cp': 2}
)

print(q2)

