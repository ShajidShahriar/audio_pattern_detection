🎤 Audio Pattern Detection Project

This project contains two Python applications for detecting repeated words or phrases in audio files using signal processing and machine learning techniques.

    V1 (app.py): A classic signal processing approach using Normalized Cross-Correlation. It's fast and great for finding exact matches.

    V2 (app_v2.py): A more advanced and robust approach using MFCCs and Dynamic Time Warping (DTW). This version is better at handling variations in speech (pitch, speed, and different speakers).

🛑 Prerequisites

Before you start, make sure you have Python 3 installed on your computer. You can check by opening your terminal or command prompt and typing python --version.
🚀 Setup & Installation (The "No Excuses" Guide)

Follow these steps exactly to avoid any errors.
Step 1: Get the Code

Download all the project files and put them in a single folder on your computer.
Step 2: Create a Virtual Environment

This is the most important step! It creates a clean, isolated space for this project so it doesn't mess with anything else on your computer.

Open your terminal or command prompt, navigate into your project folder, and run this command:

python -m venv venv

This creates a new folder named venv in your project directory.
Step 3: Activate the Virtual Environment

You have to "turn on" the clean space you just created.

    On Windows:

    .\venv\Scripts\activate

    On Mac/Linux:

    source venv/bin/activate

You'll know it's working because you'll see (venv) appear at the start of your terminal prompt.
Step 4: Install the Damn Libraries

Now, while the virtual environment is active, run this single command. It will read the requirements.txt file and install everything you need automatically.

pip install -r requirements.txt

Wait for it to finish. If you did this correctly, you will have zero dependency errors.
▶️ How to Run the Apps

Make sure your virtual environment is still active ((venv) is visible in your terminal).
To Run the V1 App (Cross-Correlation):

streamlit run app.py

To Run the V2 App (Advanced MFCC + DTW):

streamlit run app_v2.py

After running one of these commands, a new tab should automatically open in your web browser with the application running.
🎧 How to Use the Apps

The interface is simple:

    Upload Files: Use the sidebar to upload your main song (the "Target") and the short audio clip you want to find (the "Template").

        Pro Tip: For large target songs, it's best to use .mp3 files, as they are smaller and more browser-friendly.

    Tune Parameters: Use the sliders in the sidebar to adjust the detection settings.

    Analyze: Click the big "Analyze Audio" button.

    View Results: The app will show you a plot with the detections marked and a list of the timestamps where the matches were found.

That's it. Now you have everything you need to share the project without the headache.
