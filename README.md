# COSC-41001-Final-Project
## Installation Methods
### 💻 Source Code
#### Prerequisites
Python version >= 3.10.18

#### 🪜 Setup Steps
##### Step 1
-	In the terminal, clone the git repo by running: git clone https://github.com/Ozair-Rahman/COSC-41001-Final-Project.git
##### Step 2
-   In the terminal, run: `cd COSC-41001-Final-Project `
##### Step 3
-	In the terminal, run: pip install -r requirements.txt
##### Step 4
-	In the terminal, run: streamlit run –server.port 80 app.py
##### Step 5
-	In an web browser of choice, go to: http://localhost

### 🐳 Docker
#### 🪜 Setup Steps
##### Step 1
-	In the terminal, clone the git repo by running: git clone https://github.com/Ozair-Rahman/COSC-41001-Final-Project.git
##### Step 2
-   In the terminal, run: `cd COSC-41001-Final-Project `
##### Step 3
-	In the terminal, run: docker build -t lamoguy/cosc-41001-final-project .
##### Step 4
-	In the terminal, run: docker run -p 80:80 lamoguy/cosc-41001-final-project
##### Step 5
-	Navigate to a web browser of chose and in the url bar, type: http://localhost
