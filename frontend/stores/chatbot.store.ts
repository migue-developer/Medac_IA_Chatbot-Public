import { defineStore } from "pinia";
import fetchChat from "~/utils/fetcher";

type Message = {
  question: string;
  answer?: string;
};

type ChatbotStore = {
  messages: Message[];
  loading: boolean;
  dialogSpeech: boolean;
  questionObj: string;
};

export const useChatbotStore = defineStore("chatbot", {
  state: (): ChatbotStore => ({
    loading: false,
    messages: [],
    dialogSpeech: false,
    questionObj: "",
  }),
  actions: {
async sendMessage() {
  this.messages.push({
    question: this.questionObj,
  })
  this.loading = true;
  const resChat = await fetchChat('POST', "http://127.0.0.1:8000/ask", {
    question: this.questionObj,
  });  
  const lastMessage = this.messages.at(-1);
  if (lastMessage) {
    lastMessage.answer = resChat.answer;
  }
  this.loading = false;
}
  },
});
