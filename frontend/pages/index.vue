<template>
  <v-container class="ma-auto" style="width: 700px; height: 700px">
    <v-card class="d-flex flex-column h-100" flat>
      <v-card-text ref="containerRef" class="flex-grow-1 overflow-y-auto">
        <template v-if="chatbotStore.messages.length > 0">
          <v-row
            style="align-items: baseline"
            v-for="(message, index) in chatbotStore.messages"
            :key="index"
          >
            <v-col cols="12" class="pb-0">
              <v-container fluid class="d-flex justify-end">
                <v-card class="text-end" variant="tonal" rounded="xl">
                  <v-card-text>
                    {{ message.question }}
                  </v-card-text>
                </v-card>
              </v-container>
            </v-col>
            <v-col cols="12">
              <v-card class="pt-0" flat rounded="xl">
                <v-card-text>
                  <template v-if="message.answer">
                    <span class="text-body-1">{{ message.answer }}</span>
                  </template>
                  <template v-else>
                    <!-- Animación de puntos usando v-for -->
                    <div class="d-flex align-center">
                      <v-img
                        src="/logo_chat.webp"
                        height="50"
                        class="flex-0-0"
                        width="50"
                      />
                      <template v-for="(dot, idx) in 3" :key="idx">
                        <v-icon ref="dotsRefs" color="blue-darken-3">
                          mdi-circle-medium
                        </v-icon>
                      </template>
                    </div>
                  </template>
                </v-card-text>
              </v-card>
            </v-col>
          </v-row>
        </template>
        <template v-else>
          <v-img src="/logo_chat.webp" aspectRatio="1.618" />
          <p class="text-center">
            ¡Bienvenido! Habla con BiteChat y resuelve tus dudas
          </p>
        </template>
      </v-card-text>
      <v-card-actions>
        <v-form ref="form" class="w-100" @submit.prevent>
          <v-text-field
            :disabled="chatbotStore.loading"
            @keyup.enter="handleEnter"
            clearable
            @click:clear="chatbotStore.questionObj = ''"
            v-model="chatbotStore.questionObj"
            hideDetails
            variant="solo"
            placeholder="¿Qué desea preguntar?"
            class="mt-3"
            :rules="[(v) => !!v || 'Por favor, ingrese una pregunta']"
          >
            <template #prepend>
              <v-btn
                icon="mdi-microphone"
                @click="chatbotStore.dialogSpeech = true"
              />
            </template>
          </v-text-field>
        </v-form>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from "vue";
import { useAnimate } from "@vueuse/core";

const containerRef: Ref<HTMLElement | null> = ref(null);
const chatbotStore = useChatbotStore();
const dotsRefs = ref<HTMLElement[]>([]);
const form: Ref<HTMLElement | null> = ref(null);
watch(
  () => chatbotStore.loading,
  (newValue) => {
    nextTick(() => {
      if (newValue) {
        console.log(dotsRefs.value);
        dotsRefs.value.forEach((dot, index) => {
          useAnimate(
            dot,
            [
              { transform: "translateY(0px)" },
              { transform: "translateY(-15px)" },
              { transform: "translateY(0px)" },
            ],
            {
              duration: 600,
              iterations: Infinity,
              easing: "ease-in-out",
              delay: index * 200,
            }
          );
        });
      } else {
        if (!containerRef.value) return;
        containerRef.value.$el.scrollTop = containerRef.value.$el.scrollHeight;
      }
    });
  }
);

const handleEnter = async () => {
  if (!form.value) return;
  await form.value.validate();
  if (!form.value.isValid) return;

  console.log("asas");
  await chatbotStore.sendMessage();
};
</script>
