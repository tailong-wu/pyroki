import requests
import json
import time

# --- Start Timer for file loading ---
t1 = time.time()
# Load the action sequence from the JSON file
with open('action.json', 'r') as f:
    action_data = json.load(f)
ee_action_seq = action_data['ee_action_seq']
t2 = time.time()
print(f"File loading completed in {t2 - t1:.4f} seconds.")

# API endpoint URL
url = "http://localhost:8000/solve-ik/"

# Prepare the request data
request_data = {"ee_action_seq": ee_action_seq}

# --- Start Timer for request ---
t3 = time.time()

# Send the POST request
response = requests.post(url, json=request_data)

# --- End Timer for request ---
t4 = time.time()

# Check the response and print the result
if response.status_code == 200:
    # Calculate and print the elapsed time
    print(f"Request completed in {t4 - t3:.4f} seconds.")

    result = response.json()
    print("\nSuccessfully received joint angles:")
    # To make it more readable, let's pretty-print the JSON
    print(json.dumps(result, indent=4))

    # --- Start Timer for file saving ---
    t5 = time.time()
    # You can also save it to a file if needed
    with open('joint_angles_from_api.json', 'w') as f:
        json.dump(result, f, indent=4)
    t6 = time.time()
    print(f"\nSaved results to joint_angles_from_api.json in {t6 - t5:.4f} seconds.")
else:
    print(f"Error: {response.status_code}")
    print(response.text)

print(f"\nTotal client-side time: {t6 - t1:.4f} seconds.")