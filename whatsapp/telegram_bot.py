from typing import Final

# pip install python-telegram-bot
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from whatsapp.libbots import treedialog
from whatsapp.libbots.treedialog import TreeDialog

tree = TreeDialog()

print("Starting up bot...")

TOKEN: Final = "6663530744:AAE9M-ppaTbN9IIK6FB7O4Mr1K5WChtqwsU"
BOT_USERNAME: Final = "@tunante_bot"


# Lets us use the /start command
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(treedialog.INTRO)


# Lets us use the /help command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Try typing anything and I will do my best to respond!"
    )


# Lets us use the /custom command
async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "This is a custom command, you can add whatever text you want here."
    )


# Run the program
if __name__ == "__main__":
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("custom", custom_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, tree.handle_message))

    # Log all errors
    app.add_error_handler(tree.error)

    print("Polling...")
    # Run the bot
    app.run_polling(poll_interval=5)
