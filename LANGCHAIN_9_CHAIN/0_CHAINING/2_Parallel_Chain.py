from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

chatModel_1 = ChatHuggingFace(llm = hf_endpoint)
chatModel_2 = ChatHuggingFace(llm = hf_endpoint)


prompt1 = PromptTemplate(
    template="Generate a short and simple notes from the following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate a 5 MCQ from the following text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz from the following given \n {chain_notes} and {chain_quizs} ",
    input_variables=['chain_notes','chain_quizs']
)

parser =StrOutputParser()

parallel_chain = RunnableParallel({
    'chain_notes': prompt1 | chatModel_1 | parser,
    'chain_quizs': prompt2 | chatModel_2 | parser

})

mergechain = prompt3 | chatModel_1 | parser

chain = parallel_chain | mergechain

ip_text = """
Scikit-learn (also known as sklearn) is a widely-used open-source Python library for machine learning. It builds on other scientific libraries like NumPy, SciPy and Matplotlib to provide efficient tools for predictive data analysis and data mining. It offers a consistent and simple interface for a range of supervised and unsupervised learning algorithms, including classification, regression, clustering, dimensionality reduction, model selection and preprocessing.
Wide Range of Algorithms: Scikit-learn provides access to a rich selection of algorithms for classification, regression, clustering and dimensionality reduction.
Easy to Use and Understand: Clean API design and documentation make it suitable for both beginners and professionals.
Interoperability: Works seamlessly with NumPy, Pandas, Matplotlib and other Python libraries.
Feature Engineering and Evaluation Tools: Includes preprocessing utilities, pipelines and model evaluation metrics.
Production-Ready: Optimized for performance and scalable to large datasets.Supervised learning involves training models on labeled data to make predictions. Scikit-learn offers a variety of algorithms such as Linear Regression, SVM, Decision Trees and Random Forests to solve classification and regression problems.Classification Models in Scikit-Learn
Linear Regression using sklearn
Multiple Linear Regression With scikit-learn
SVM and Kernel SVM with Scikit-Learn
RBF SVM with Scikit Learn
Decision Tree Classifiers with Scikit-Learn
Decision Tree Regression using sklearn
Random Forest Classifier using Scikit-learn
KNN classifier using Scikit-Learn
Gaussian Naive Bayes using Sklearn
Stochastic Gradient Descent Regressor using Scikit-learn
Unsupervised Learning with Scikit-Learn
In unsupervised learning, models are trained on unlabeled data to find hidden patterns or groupings. Explore clustering techniques like K-Means and DBSCAN and dimensionality reduction methods like PCA and manifold learning.
K-Means clustering using Scikit Learn
DBSCAN algorithm using Sklearn
PCA with scikit-learn
Hierarchical Clustering with Scikit-Learn
Gaussian Mixture Models (GMM) in Scikit Learn
Manifold Learning methods in Scikit Learn

"""

result = chain.invoke({'text':ip_text})

print(result)





























