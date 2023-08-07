from telegram import  InlineKeyboardMarkup, Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler,ConversationHandler, filters, ContextTypes
import config
import datetime
import predict
import train
import trade
import json

TPSL, LVRG, AMOUNT = range(3)

main_keyboard = [[InlineKeyboardButton("Start New Trading", callback_data= 'm1')],
                 [InlineKeyboardButton("Stop From Next Trading", callback_data= 'm2')],
                 [InlineKeyboardButton("Get Daily ROI", callback_data= 'm3')],
                 [InlineKeyboardButton("Update Model with Recent Data", callback_data= 'm4')]]

trading_keyboard = [[InlineKeyboardButton("Confirm from Next Trading", callback_data= 'm1_1')],
                    [InlineKeyboardButton("Predict Next hour price", callback_data= 'm1_2')],
                    [InlineKeyboardButton("Modify Trading Options", callback_data= 'm1_3')],
                    [InlineKeyboardButton("<- previous ", callback_data ='main')]]

modify_keyboard = [[InlineKeyboardButton("Modify TP/SL", callback_data= 'm1_3_1')],
                   [InlineKeyboardButton("Modify Leverage options", callback_data= 'm1_3_2')],
                   [InlineKeyboardButton("Modify Single Trading amount", callback_data= 'm1_3_3')],
                   [InlineKeyboardButton("<- previous ", callback_data ='m1_3')]]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Please choose your options:",
                                    reply_markup=InlineKeyboardMarkup(main_keyboard))

#main
async def main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("Please choose your options:",
                                  reply_markup=InlineKeyboardMarkup(main_keyboard))



#m1
async def trading_menu(update:Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("Please choose your options:",
                                  reply_markup=InlineKeyboardMarkup(trading_keyboard))

#m1_1
async def confirm_menu(update:Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    ######
    #Reserve
    calcul_text, _ = predict.main()
    trade.main()

    ###
    current = datetime.datetime.now()
    next = current.hour + 1
    await query.message.edit_text(text="Current Time: " + current.strftime("%H:%M:%S")
                                       + f"Your trade will be starting at {next}",
                                  reply_markup=InlineKeyboardMarkup(main_keyboard))

#m1_2
async def prediction(update:Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    calcul_text, _ = predict.main()
    current = datetime.datetime.now()
    next = current.hour + 1
    await query.edit_message_text(f"\nPrediction at {next}:00\n\n{calcul_text}",
                                  reply_markup=InlineKeyboardMarkup(main_keyboard))



#m1_3
async def modifying_menu(update:Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    txt = ""
    with open("trade_info.json", 'r') as f:
        info = json.load(f)
    for k in info:
        txt += f"{k}: {info[k]}\n"

    await query.message.edit_text(f"Current Settings:\n {txt}",
                                  reply_markup=InlineKeyboardMarkup(modify_keyboard))

#m1_3_1
async def modify_tpsl(update:Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    #await query.answer()
    await query.edit_message_text("Choose your TP/SL, if +-20%, type '20' or you don't want any, type 'None'")
    return TPSL

async def tpsl_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # SET TPSL
    text = update.message.text
    if int(text) >= 5 and int(text) <= 50:
        modify_trade_info(component="tpsl", val=int(text))
        await update.message.reply_text(f"Your TP/SL rate has been set to {text}%",
                                        reply_markup=InlineKeyboardMarkup(main_keyboard))
        return ConversationHandler.END


    else:
        await update.message.reply_text("ERROR, your TP/SL rate is too low or high, please choose between 5 ~ 50")
        return TPSL

#m1_3_2

async def modify_leverage(update:Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("Choose your Leverage, if 20x, type '20'")

    return LVRG

async def lvrg_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # SET Leverage
    text = update.message.text
    if int(text) <= 75:
        modify_trade_info(component="leverage", val=int(text))
        await update.message.reply_text(f"Your leverage has been set to {text}X",
                                        reply_markup=InlineKeyboardMarkup(main_keyboard))
        return ConversationHandler.END


    else:
        await update.message.reply_text("ERROR, your leverage is too high, please choose lower than 75")
        return LVRG

#m1_3_3

async def modify_amount(update:Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("Choose your single bidding amount based on the total balance, if 20%, type '20'")

    return AMOUNT

async def amount_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # SET Amount
    text = update.message.text
    # Get total balance and calculate the percentage
    modify_trade_info(component="amount", val=int(text))
    if type(text) == int:
        await update.message.reply_text(f"Your single bidding amount has been set to ${text}",
                                        reply_markup=InlineKeyboardMarkup(main_keyboard))
        return ConversationHandler.END

    else:
        await update.message.reply_text("ERROR: Please type again")
        return AMOUNT


def modify_trade_info(component, val):
    with open("trade_info.json", 'r') as f:
        info = json.load(f)
    info[component] = val
    with open("trade_info.json", 'w') as f:
        json.dump(info,f, indent=4)



#m2
async def canceling_menu(update:Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("Please choose your options:",
                                  reply_markup=InlineKeyboardMarkup(trading_keyboard))
#m3
async def ROI_menu(update:Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("Please choose your options:",
                                  reply_markup=InlineKeyboardMarkup(trading_keyboard))
#m4
async def update_menu(update:Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.edit_message_text("Please Wait, it will take few minutes")
    train.main()
    await update.callback_query.edit_message_text("Your model has been successfully updated!",
                                                  reply_markup=InlineKeyboardMarkup(main_keyboard))





async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi, I'm AI-based Auto-Crypto Trading bot.\n Use /start to test this bot.")

# async def error(update: Update, context:ContextTypes.DEFAULT_TYPE):
#     print(f'Update {update} caused error {context.error}')


async def kill(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    await update.message.reply_text(
        "Bye! I hope we can talk again some day.", reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END

def main():
    application = Application.builder().token(config.key).build()

    application.add_handler(CommandHandler('start',start))
    application.add_handler(CallbackQueryHandler(main_menu, pattern ='main'))

    application.add_handler(CallbackQueryHandler(trading_menu, pattern='^m1$'))
    application.add_handler(CallbackQueryHandler(confirm_menu, pattern='^m1_1$'))

    application.add_handler(CallbackQueryHandler(prediction, pattern='^m1_2$'))

    application.add_handler(CallbackQueryHandler(modifying_menu, pattern='^m1_3$'))
    #
    application.add_handler(CallbackQueryHandler(modify_tpsl, pattern= '^m1_3_1$'))
    application.add_handler(CallbackQueryHandler(modify_leverage, pattern= '^m1_3_2$'))
    application.add_handler(CallbackQueryHandler(modify_amount, pattern='^m1_3_3$'))

    application.add_handler(CallbackQueryHandler(canceling_menu, pattern='^m2$'))
    application.add_handler(CallbackQueryHandler(ROI_menu, pattern='^m3$'))
    application.add_handler(CallbackQueryHandler(update_menu, pattern='^m4$'))

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.TEXT, modify_tpsl),
                      MessageHandler(filters.TEXT, modify_leverage),
                      MessageHandler(filters.TEXT, modify_amount)],
        states={
            TPSL: [MessageHandler(filters.TEXT, tpsl_choice)],
            LVRG: [MessageHandler(filters.TEXT,lvrg_choice)],
            AMOUNT: [MessageHandler(filters.TEXT,amount_choice)]
        },
        fallbacks = [MessageHandler(filters.TEXT,kill)]
    )

    # application.add_error_handler(error)
    application.add_handler(conv_handler)

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()