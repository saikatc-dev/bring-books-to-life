# Bring Books to Life

## What is this project?

Bring Books to Life lets you have a conversation with a character from a book. It currently comes with Sherlock Holmes and Professor James Moriarty, his adversary. Ask either character questions and they reply in character, using details from the book rather than making up book-related facts.

You can use the chat in a web page or in a terminal. Choose a character in the web app, or pass their name when starting a terminal chat. The web page also lets you upload a `.txt` file so the selected character can answer using a different text while keeping their personality.

## What it can do

- **Chat as a book character:** The character responds in their own voice.
- **Look up details in the book:** When it needs information about the story, the AI searches relevant passages before replying. If the text does not contain an answer, it should say so rather than inventing one.
- **Remember the current conversation:** It can refer to earlier messages while the conversation is open. This memory is temporary and is cleared when you reset or restart the chat.
- **Use your own text:** Upload a plain text `.txt` file in the web app. This changes the source text for the current character, not the character's personality.
- **Run book search locally:** The text-search model runs on your computer. It downloads a small model the first time you use it; no separate embedding API key is needed.

## What you need

- Python 3.12
- A Groq API key for AI replies. Create one at [Groq Console](https://console.groq.com/keys).
- An internet connection for the first setup, to install packages and download the local text-search model.

## Set up on Windows

Open PowerShell in this project folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open the new `.env` file and replace `your-free-groq-key-here` with your Groq API key. Keep the key private.

If PowerShell prevents activation, you can still use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Start the project

For the web chat:

```powershell
streamlit run app.py
```

For a terminal chat:

```powershell
python run.py
```

To chat with Professor Moriarty instead:

```powershell
python run.py moriarty
```

To run a sample conversation in the terminal:

```powershell
python run.py --demo
```

To check that book search works without using a Groq API key:

```powershell
python test_offline.py
```

The first offline test run may take longer because it downloads the text-search model.

## How it works, in simple terms

1. The project splits the book into small passages and prepares them for searching on your computer.
2. You ask a question.
3. The AI decides whether it needs to look up something in the book, then searches for relevant passages.
4. It uses those passages and the character's description to write a reply.

## Project files

| File or folder | What it is for |
|---|---|
| `app.py` | The web chat and text-file uploader |
| `run.py` | The terminal chat and sample conversation |
| `agent.py` | Connects the AI, character instructions, book search, and conversation memory |
| `personas.py` | Defines the available characters and their speaking styles |
| `rag.py` | Prepares book text and searches for relevant passages |
| `tools.py` | Gives the AI a way to search the book |
| `sources/` | Book text files; includes the Sherlock Holmes source |
| `test_offline.py` | Checks book search without making AI chat requests |
| `requirements.txt` | The Python packages needed to run the project |

To add another character, add their book text under `sources/` and define the character in `personas.py`.
