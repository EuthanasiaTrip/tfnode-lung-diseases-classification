const fs = require('fs');

class Logger {
    constructor() {
        this.logFolder = "./logs";
        this.logFile = this.logFolder + '/' + new Date().toISOString()
            .replace(/T/, ' ')
            .replace(/\..+/, '')
            .replace(':', '-')
            .replace(':', '-') + ".txt";
        this._setFolder();
        this._createLogFile();
    }

    _setFolder(){
        if(!fs.existsSync(this.logFolder)){
            fs.mkdirSync(this.logFolder);
        }
    }

    _createLogFile(){      
       fs.writeFileSync(this.logFile, `Logs for ${new Date()}`,{
           flag: 'a'
       } ,err => {
           if(err) {
               console.log(err);
               return;
           }
       });
    }

    /**
     * @private
     * @param {string} msg  - message to format
     * @returns formatted message with timestamp
     */
    _formatMsg(msg) {
        const timeStamp = new Date().toISOString().replace(/T/, ' ').replace(/\..+/, '');
        return "[" + timeStamp + "]" + msg;
    }

    log(msg, needFormat = true){        
        const formattedMsg = needFormat ? this._formatMsg(JSON.stringify(msg)) : JSON.stringify(msg);
        console.log(formattedMsg);
        fs.appendFileSync(this.logFile, `\n${formattedMsg}`, err => {
            if(err){
                console.log(err);
            }
        });
    }
}

module.exports = new Logger();