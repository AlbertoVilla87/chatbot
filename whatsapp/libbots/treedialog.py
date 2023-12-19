from telegram import Update
from telegram.ext import ContextTypes

INTRO = "Hola mi nombre es Alberto Bot, gracias a mi creador Alberto, soy capaz de contestar automáticamente y tunantemente 🤣"
EXPLANATION = "Vaya etapa más bonita que habéis vivido, he podido leer todas vuestras conversaciones. Qué cosas más bonitas os habéis dicho!!! Quieres ver las cosas que dijiste a mi creador?"


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
            text = "Me alegra que te haya gustado, querrías ver ahora lo que él te ha dicho"
            await update.message.reply_text(text)

        elif (text == "si") & (self.tree_id == 3):
            self.tree_id = 4
            photo = open("whatsapp/data/pictures/replay.png", "rb")
            await update.message.reply_photo(photo)
            photo.close()

        elif self.tree_id == 4:
            self.tree_id = 5
            text = "Pues esto no es todo porque también habéis vivido grande momentos, los quieres ver?"
            await update.message.reply_text(text)

        elif self.tree_id == 5:
            self.tree_id = 6
            video_path = "whatsapp/data/pictures/love.mp4"
            await update.message.reply_video(video_path)

        elif self.tree_id == 6:
            self.tree_id = 7
            text = "Lo mejor de todo, es que esto solo el principio, os queda mucho futuro por delante. Pero es verdad que es muy bonito recordar aquellos días especiales. Así que para que siempre recuerdes este día, hay una sorpresa guardada en el frutero."
            await update.message.reply_text(text)

        else:
            await update.message.reply_text("Respuesta incorrecta preciosa")

    # Log errors
    async def error(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        print(f"Update {update} caused error {context.error}")
