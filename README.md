![ai-resume-parser](https://socialify.git.ci/NiteeshL/ai-resume-parser/image?custom_description=Resume+extraction+and+evaluation+tool+powered+by+Generative+AI%2C+which+processes+resumes+to+extract+key+information+and+rank+candidates+based+on+relevance+and+role+fit.+It+leverages+AI+models+to+ensure+high+accuracy+and+scalability+in+batch+processing.&description=1&language=1&name=1&owner=1&theme=Light)
# Resume Processor

Resume Processor is a Python-based application that extracts and processes information from resumes placed inside a GOOGLE DRIVE folder to reduce the hassle 😉. It uses the Gemini API to extract key details and calculate experience scores based on keyword matches.
Alternatively, it can also be used to process resumes present in the machine.

## Features

- Google drive integration (drive folder doesn't have to be publicly accessible 😉)
- Batch Processing of multiple resumes with timely output
- Extracts text from PDF resumes
- Queries the Gemini API to extract key details from resumes
- Calculates experience scores based on keyword matches
- Provides a web interface for processing resumes

## Project Structure

```
/c:/Users/niteesh/Downloads/Projects/ai-resume-parser/
├── templates/
│   └── index.html
├── credentials.json
├── resume.py
├── README.md
├── requirements.txt
└── .env
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/NiteeshL/ai-resume-parser.git
    cd ai-resume-parser
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up Google Drive API credentials:
    - Follow the instructions to create a project and enable the Google Drive API: [Google Drive API Quickstart](https://developers.google.com/drive/api/v3/quickstart/python)
    - Download the `credentials.json` file and place it in the project root directory.

5. Create a Gemini API key:
    - Sign up for an account at [Gemini](https://gemini.com) if you don't have one.
    - Navigate to the API section and create a new API key.
    - Create a `.env` file in the project root directory and add your Gemini API key:
        ```dotenv
        GEMINI_API_KEY=your_actual_api_key_here
        ```

## Usage

1. Run the Flask application:
    ```sh
    python resume.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

3. Enter the Google Drive Folder ID containing the resumes in the input field and click "Process Resumes". The application can process a large number of resumes efficiently.

4. The application will process the resumes and save the extracted data to `processed_resumes_with_scores.xlsx`.

## Images and videos
https://github.com/user-attachments/assets/6071ff1b-3151-426d-9088-c241bd0ed499


![web page](https://github.com/NiteeshL/ai-resume-parser/blob/main/image.png)


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF text extraction
- [Google Drive API](https://developers.google.com/drive) for file management
- [Gemini API](https://gemini.com) for resume data extraction

## Contact

For any questions or inquiries, please contact [niteeshleela@gmail.com](mailto:niteeshleela@gmail.com).
