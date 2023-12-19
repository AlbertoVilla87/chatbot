from telegram import Update
from telegram.ext import ContextTypes

INTRO = "****"
EXPLANATION = "*****"


class TreeDialog:
    def __init__(self):
        """_summary_"""
        self.tree_id = 0

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Get basic info of the incoming message
        message_type: str = update.message.chat.type
        text: str = update.message.text
        text = text.lower()

        # Print a log for debugging
        print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

        if text == "hola":
            self.tree_id = 1
            text = EXPLANATION
            await update.message.reply_text(text)

        elif (text == "si") & (self.tree_id == 1):
            self.tree_id = 2
            photo = open("whatsapp/data/pictures/ask.png", "rb")
            await update.message.reply_photo(photo)
            photo.close()

        elif self.tree_id == 2:
            self.tree_id = 3
            text = "*******"
            await update.message.reply_text(text)

        elif (text == "si") & (self.tree_id == 3):
            self.tree_id = 4
            photo = open("whatsapp/data/pictures/replay.png", "rb")
            await update.message.reply_photo(photo)
            photo.close()

        elif self.tree_id == 4:
            self.tree_id = 5
            text = "*****"
            await update.message.reply_text(text)

        elif self.tree_id == 5:
            self.tree_id = 6
            video_path = "whatsapp/data/pictures/love.mp4"
            await update.message.reply_video(video_path)

        elif self.tree_id == 6:
            self.tree_id = 7
            text = "*****"
            await update.message.reply_text(text)

        else:
            await update.message.reply_text("Respuesta incorrecta")

    # Log errors
    async def error(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        print(f"Update {update} caused error {context.error}")
