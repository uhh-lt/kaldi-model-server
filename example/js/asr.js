var scrollBottom = true;
var scrollChatAreaBottom = true;

/*Events: speech recognition feedback*/

function renderUtterance(jsonEvent) {
	return '<span>'+jsonEvent.speaker+':</span> '+jsonEvent.utterance
}

function addUtterance(jsonEvent) {
	$('#chat-area').append('<p>'+renderUtterance(jsonEvent)+' </p>')
	if(scrollChatAreaBottom)
		document.getElementById('chat-area').scrollTop = document.getElementById('chat-area').scrollHeight;
}

function replaceLastUtterance(jsonEvent) {
	$('#chat-area p:last').html(renderUtterance(jsonEvent))
	if(scrollChatAreaBottom)
		document.getElementById('chat-area').scrollTop = document.getElementById('chat-area').scrollHeight;
}

/*Dispatch events from EventSource*/

var source = new EventSource('/stream');
var utts = [];

var startNewUtt = true;

source.onmessage = function (event) {
  console.log((JSON.parse(event.data));
	jsonEvent = JSON.parse(event.data);

	if (jsonEvent.handle == 'partialUtterance')
	{
    if(startNewUtt) {
  		utts.push(jsonEvent.utterance);
	  	addUtterance(jsonEvent);
    }else{ 
      utts.pop();
      utts.push(jsonEvent.utterance);
      replaceLastUtterance(jsonEvent);
    }
	}
	else if (jsonEvent.handle == 'completeUtterance')
	{
		utts.pop();
		utts.push(jsonEvent.utterance);
		replaceLastUtterance(jsonEvent);
    startNewUtt = true;
	}	else if (jsonEvent.handle == 'reset')
	{
		reset();
	}
};
