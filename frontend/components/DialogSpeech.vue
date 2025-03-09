<template>
  <v-overlay v-model="chatbotStore.dialogSpeech" class="d-flex align-center justify-center">
    <v-card  flat variant="text">
      <v-card-text class="mx-auto pb-10">
        <div class="d-flex justify-center ga-3" style="height: 55px;">
          <v-divider
            v-for="(_, idx) in 7"
            :key="idx"
            ref="bars"
            color="white"
            class="opacity-100 mr-1"
            vertical
            thickness="5px"
          ></v-divider>
        </div>
      </v-card-text>
    </v-card>
  </v-overlay>
</template>

<script setup lang="ts">
import { useAnimate, useSpeechRecognition } from "@vueuse/core";
import { ref, watch, nextTick } from "vue";

const chatbotStore = useChatbotStore();
const bars = ref<HTMLElement[]>([]);

const { isSupported, result, start, stop, isListening } = useSpeechRecognition({
  lang: "es-ES",
  continuous: false,
  interimResults: false,
});

const animateBar = (el: HTMLElement, delay: number) => {
  useAnimate(
    el,
    [
      { transform: "scaleY(1)", offset: 0 },
      { transform: "scaleY(2.5)", offset: 0.4 },
      { transform: "scaleY(1.5)", offset: 0.6 },
      { transform: "scaleY(1)", offset: 1 },
    ],
    {
      duration: 800,
      iterations: Infinity,
      easing: "ease-in-out",
      delay,
    }
  );
};

watch(
  () => result.value,
  (res) => {
    if (res) {
        chatbotStore.dialogSpeech = false;
        chatbotStore.questionObj = res;        
    }
  }
);

watch(
  () => chatbotStore.dialogSpeech,
  (value) => {
    nextTick(() => {
      if (value && isSupported.value) {
        start();
        bars.value.forEach((bar, idx) => animateBar(bar, idx * 100));
      }
    });
  },
  { immediate: true }
);
</script>
