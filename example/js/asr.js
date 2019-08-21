let scrollBottom = true;
let scrollChatAreaBottom = true;

/*Events: speech recognition feedback*/

function renderUtterance(jsonEvent) {
    return '<span class="speaker">' + jsonEvent.speaker + ':</span> ' + jsonEvent.utterance;
}

function renderUtteranceWithConfidence(jsonEvent) {
    let result = '<span class="speaker">' + jsonEvent.speaker + ':</span> ';
    let text = jsonEvent.utterance.split(" ");

    for(let i = 0; i < text.length; i++) {
        let word = text[i];
        let conf = jsonEvent.confidences[i];
        result += '<span style="color:rgba(0,0,0,' + Math.max(conf * conf, 0.1) + ');">' + word + '</span> '
    }

    console.log(result);
    return result.trim();
}

function addUtterance(jsonEvent) {
    $('#chat-area').append('<p>' + renderUtterance(jsonEvent) + ' </p>');
    if (scrollChatAreaBottom)
        document.getElementById('chat-area').scrollTop = document.getElementById('chat-area').scrollHeight;
}

function replaceLastUtterance(jsonEvent, renderConfidence) {
    if(renderConfidence) {
        $('#chat-area p:last').html(renderUtteranceWithConfidence(jsonEvent));
    } else {
        $('#chat-area p:last').html(renderUtterance(jsonEvent));
    }
    if (scrollChatAreaBottom)
        document.getElementById('chat-area').scrollTop = document.getElementById('chat-area').scrollHeight;
}

/*Dispatch events from EventSource*/

let source = new EventSource('/stream');
let utts = [];

let startNewUtt = true;

source.onmessage = function (event) {
    console.log(event.data);
    let jsonEvent = JSON.parse(event.data);

    if (jsonEvent.handle === 'partialUtterance') {
        if (startNewUtt) {
            utts.push(jsonEvent.utterance);
            addUtterance(jsonEvent);
            startNewUtt = false;
        } else {
            utts.pop();
            utts.push(jsonEvent.utterance);
            replaceLastUtterance(jsonEvent, false);
        }
    } else if (jsonEvent.handle === 'completeUtterance') {
        utts.pop();
        utts.push(jsonEvent.utterance);
        replaceLastUtterance(jsonEvent, true);
        startNewUtt = true;
    } else if (jsonEvent.handle === 'reset') {
        reset();
    }
};
