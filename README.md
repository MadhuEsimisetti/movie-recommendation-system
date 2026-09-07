\# 🎬 Movie Recommendation System



A content-based movie recommendation system built using Python, Pandas, Scikit-learn, and Streamlit.



The system recommends movies similar to a movie selected by the user.



\## 🚀 Live Demo



Run the application locally using Streamlit.



\## 🧠 How It Works



The recommendation system uses \*\*content-based filtering\*\*.



Movie information such as:



\- Genres

\- Keywords

\- Top 3 cast members

\- Director



is combined into a single `tags` feature.



\### 1. Feature Engineering



The selected movie features are combined into one text representation.



Example:



```text

Action Adventure Sci-Fi space alien Sam Worthington Zoe Saldana James Cameron 



2\. TF-IDF



TF-IDF (Term Frequency-Inverse Document Frequency) converts the movie tags into numerical vectors.



This allows the machine learning model to represent each movie based on the importance of its words.



3\. Cosine Similarity



Cosine similarity is used to measure how similar two movie vectors are.



Movies with higher similarity scores are considered more relevant.



4\. Recommendations



For a selected movie, the system calculates similarity with other movies, sorts the results by similarity score, and returns the top 5 recommendations.



🛠️ Technologies Used

Python

Pandas

Scikit-learn

Streamlit

TF-IDF

Cosine Similarity


## 📂 Project Structure

```text
movie-recommendation-system/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
