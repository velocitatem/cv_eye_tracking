import pickle
import os
import numpy as np
from collections import defaultdict
import face_recognition  # You might need to install this: pip install face_recognition
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load existing observations
observations = []
if os.path.exists("observations.pkl"):
    with open("observations.pkl", "rb") as f:
        observations = pickle.load(f)
print(f"Loaded {len(observations)} observations.")

# Extract face encodings for each observation
face_encodings = []
attention_scores = []
timestamps = []  # Assuming observations are in chronological order
for i, (face_image, attention) in enumerate(observations):
    # Get face encoding
    encoding = face_recognition.face_encodings(face_image)
    if len(encoding) > 0:  # Make sure a face was detected
        face_encodings.append(encoding[0])
        attention_scores.append(attention)
        timestamps.append(i)  # Use index as timestamp if no actual timestamp exists
    else:
        print(f"No face detected in observation {i}")

# Cluster faces to identify unique people using DBSCAN
# This is more robust than comparing each face to every other face
face_encodings_array = np.array(face_encodings)
clustering = DBSCAN(eps=0.6, min_samples=3, metric="euclidean").fit(face_encodings_array)
face_ids = clustering.labels_

# Map each face to a person ID
person_attention_series = defaultdict(list)
person_timestamps = defaultdict(list)
person_images = defaultdict(list)

for i, face_id in enumerate(face_ids):
    if face_id != -1:  # Skip outliers labeled as -1 by DBSCAN
        person_attention_series[face_id].append(attention_scores[i])
        person_timestamps[face_id].append(timestamps[i])
        person_images[face_id].append(observations[timestamps[i]][0])  # Store original image

# Calculate average attention for each person
person_avg_attention = {}
for person_id, attention_values in person_attention_series.items():
    person_avg_attention[person_id] = np.mean(attention_values)

# Sort people by average attention
sorted_people = sorted(person_avg_attention.items(), key=lambda x: x[1], reverse=True)

# Print results
print(f"Identified {len(person_avg_attention)} unique people")
print("\nAttention Ranking:")
for rank, (person_id, avg_attention) in enumerate(sorted_people, 1):
    num_appearances = len(person_attention_series[person_id])
    print(f"Person {person_id}: Avg Attention = {avg_attention:.2f} (Appeared in {num_appearances} frames)")

# Save the processed data
results = {
    "person_attention_series": dict(person_attention_series),
    "person_timestamps": dict(person_timestamps),
    "person_avg_attention": person_avg_attention,
    "person_images": dict(person_images)
}

with open("attention_analysis_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nResults saved to 'attention_analysis_results.pkl'")

# Plot attention over time for each person
plt.figure(figsize=(12, 8))
for person_id, timestamps in person_timestamps.items():
    attention_values = person_attention_series[person_id]
    plt.plot(timestamps, attention_values, 'o-', label=f"Person {person_id}")

plt.xlabel("Time")
plt.ylabel("Attention Score")
plt.title("Attention Scores Through Time")
plt.legend()
plt.grid(True)
plt.savefig("attention_time_series.png")

# Optional: Create a visualization that shows each person with their average attention
def visualize_people_with_scores():
    num_people = len(person_avg_attention)
    cols = min(5, num_people)
    rows = (num_people + cols - 1) // cols

    plt.figure(figsize=(15, 3 * rows))
    for i, person_id in enumerate(sorted_people):
        person_id = person_id[0]  # Extract person_id from the tuple

        # Select a representative image for this person
        representative_img = person_images[person_id][len(person_images[person_id])//2]

        plt.subplot(rows, cols, i+1)
        plt.imshow(representative_img)
        plt.title(f"Person {person_id}\nAttn: {person_avg_attention[person_id]:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("people_attention_scores.png")
    print("Visualization saved to 'people_attention_scores.png'")

visualize_people_with_scores()

# Additional analysis: attention variance for each person
person_attention_variance = {}
for person_id, attention_values in person_attention_series.items():
    if len(attention_values) > 1:  # Need at least 2 values for variance
        person_attention_variance[person_id] = np.var(attention_values)

# People with the most variable attention
if person_attention_variance:
    most_variable = sorted(person_attention_variance.items(), key=lambda x: x[1], reverse=True)
    print("\nPeople with most variable attention:")
    for person_id, variance in most_variable[:3]:
        print(f"Person {person_id}: Variance = {variance:.4f}")
