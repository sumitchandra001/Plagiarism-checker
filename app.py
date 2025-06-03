import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(_file, encoding='utf-8').read()
                 for _file in student_files]


def vectorize(Text): return TfidfVectorizer().fit_transform(Text).toarray()
def similarity(doc1, doc2): return cosine_similarity([doc1, doc2])


vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()


def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results


print("\n🔍 Plagiarism Check Results:\n")
print("------------------------------------------------------------")
print(f"{'File 1':<20} {'File 2':<20} {'Similarity Score'}")
print("------------------------------------------------------------")

for file1, file2, score in sorted(check_plagiarism(), key=lambda x: x[2], reverse=True):
    status = ""
    if score > 0.75:
        status = "⚠ High similarity! Likely copied."
    elif score > 0.5:
        status = "⚠ Moderate similarity. Needs review."
    elif score > 0.3:
        status = "🔎 Some similarity. Possibly coincidental."
    else:
        status = "✅ Low similarity. Likely original."

    print(f"{file1:<20} {file2:<20} {score:.2f} {status}")
