# AI Proposal Assistant

This is a Slack-based AI assistant that helps a business consultant turn client discovery call transcripts into a professional proposal document.

## Setup Instructions

1. **Clone the repository** and ensure you have Python 3.11+.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory with the following variables:
   ```env
   # Slack Configuration
   SLACK_BOT_TOKEN=xoxb-your-bot-token
   SLACK_SIGNING_SECRET=your-signing-secret
   SLACK_BOT_USER_ID=your-bot-user-id
   VERIFY_SLACK_SIGNATURE=true

   # LLM API
   GROQ_API_KEY=your-groq-api-key
   LLM_MODEL=llama-3.1-70b-versatile
   
   # Qdrant Vector DB
   QDRANT_URL=data/qdrant_db # Local persistent by default
   QDRANT_API_KEY= # Optional, if using managed Qdrant
   
   # Optional configurations
   WRITER_TEMPERATURE=0.4
   PROPOSALS_DIR=data/proposals
   ```

## Running the Application

1. **Ingest Seed Proposals:**
   Before running the API, ingest the sample past proposals into the vector database (this is also done automatically if the collection is empty, but can be done manually):
   ```bash
   python scripts/ingest_proposals.py
   ```


2. **Start the API Server:**
   Start the FastAPI app using Uvicorn:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

3. **Expose to Slack:**
   Use ngrok to expose your local server so Slack can send events to it:
   ```bash
   ngrok http 8000
   ```
   *Update your Slack App's Event Subscriptions Request URL to `https://<ngrok-url>/slack/events`.*

## How to Use

1. Upload a `.txt` transcript file in a DM to the bot or tag the bot in a channel.
2. The bot will acknowledge the upload, process the transcript, extract information, perform vector search on past proposals, and generate a tailored DOCX proposal draft.
3. Once generated, the bot will deliver the draft DOCX to Slack.
4. You can reply with questions about the proposal/client context.
5. You can reply with revision requests (e.g. "make the timeline section more detailed"). The system will update just that section and return a new DOCX.
6. Reply with "approve" to finalize the proposal!ubmission.  Be ready to walk through your code and explain your decisions.

