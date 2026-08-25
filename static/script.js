const chatBox = document.getElementById("chat-box");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const typingIndicator = document.getElementById("typing-indicator");


function addMessage(message, sender) {

    const messageDiv = document.createElement("div");

    messageDiv.classList.add(
        "message",
        sender === "user"
            ? "user-message"
            : "bot-message"
    );


    const avatar = document.createElement("div");

    avatar.classList.add("avatar");

    avatar.textContent =
        sender === "user"
            ? "👤"
            : "🤖";


    const bubble = document.createElement("div");

    bubble.classList.add("bubble");

    bubble.textContent = message;


    messageDiv.appendChild(avatar);
    messageDiv.appendChild(bubble);

    chatBox.appendChild(messageDiv);

    chatBox.scrollTop = chatBox.scrollHeight;
}


function showTyping() {
    typingIndicator.classList.remove("hidden");
}


function hideTyping() {
    typingIndicator.classList.add("hidden");
}


async function sendMessage() {

    const message = messageInput.value.trim();

    if (!message) {
        return;
    }


    // Display user message
    addMessage(message, "user");

    messageInput.value = "";

    sendButton.disabled = true;

    showTyping();


    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: message
            })

        });


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.error || "Something went wrong."
            );

        }


        // Display chatbot response
        addMessage(
            data.response || "No response received.",
            "bot"
        );


    } catch (error) {

        addMessage(
            "Sorry, I couldn't connect to the chatbot. Please try again.",
            "bot"
        );

        console.error(error);

    } finally {

        hideTyping();

        sendButton.disabled = false;

        messageInput.focus();

    }
}


/* Send button */
sendButton.addEventListener(
    "click",
    sendMessage
);


/* Enter key */
messageInput.addEventListener(
    "keydown",
    function(event) {

        if (event.key === "Enter") {

            event.preventDefault();

            sendMessage();

        }

    }
);


/* Focus input when page loads */
messageInput.focus();
