# Resume-Paraphrasing
A specialized microservice for resume text enhancement with multiple paraphrasing styles using locally hosted LLM Models. Implemented parallel processing, caching mechanisms, and fallback strategies for reliable operation.

🔧 Setting Up LLaMA 3.1 (8B) with Ollama
To run this project with the LLaMA 3.1 - 8B model using Ollama, follow the steps below:

Step 1: Install Ollama
If Ollama is not installed on your system:

Windows / macOS / Linux:
Download and install it from the official website:
👉 https://ollama.com/download

Step 2: Download LLaMA 3.1 (8B) Model
After installing Ollama, open your terminal and run:

Copy code: ollama pull llama3:8b
This will download the LLaMA 3.1 8B model to your system.

If you want a smaller version, use:
Copy code: ollama pull llama3:instruct

✅ Step 3: Run the Model (Optional- To start a local chat with the model)
Copy code: ollama run llama3:8b
