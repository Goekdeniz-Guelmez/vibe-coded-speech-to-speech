# Runn that motherfucker

```shell
   cd v2/backend
   pip install -r requirements.txt
   
   # Zusätzliche Abhängigkeiten installieren
   pip install uvicorn websockets
   
   # Backend starten
   uvicorn main:app --host 0.0.0.0 --port 8000
```



```shell
   cd v2/frontend
   npm install  # Falls noch nicht gemacht
   npm run dev  # Startet den Entwicklungsserver
```

---

# Just UI

```javascript
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MicOff, VolumeX, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

export default function SpeechToSpeechUI() {
  const [isReady, setIsReady] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [audioLevel, setAudioLevel] = useState(1);
  const [micMuted, setMicMuted] = useState(false);
  const [outputMuted, setOutputMuted] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [persona, setPersona] = useState("base");
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful AI assistant.");
  const [voice, setVoice] = useState("default");
  const [model, setModel] = useState("gpt-4");
  const [grainPosition, setGrainPosition] = useState("0% 0%");
  const [speakerState, setSpeakerState] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setIsReady(true), 2000);
    const match = window.matchMedia("(prefers-color-scheme: dark)");
    setIsDarkMode(match.matches);
    match.addEventListener("change", (e) => setIsDarkMode(e.matches));
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      const level = micMuted ? 1 : 1 + Math.random() * 0.1;
      setAudioLevel(level);

      const randX = Math.floor(Math.random() * 100);
      const randY = Math.floor(Math.random() * 100);
      setGrainPosition(`${randX}% ${randY}%`);
    }, 250);
    return () => clearInterval(interval);
  }, [micMuted]);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/realtime-sts");

    ws.onopen = () => {
      console.log("🌐 WebSocket connected");
    };

    ws.onmessage = async (event) => {
      try {
        if (typeof event.data === "string") {
          const msg = JSON.parse(event.data);
          if (msg.type === "state") {
            setSpeakerState(msg.value); // 0 = idle, 1 = user, 2 = assistant
          }
        } else {
          // Handle audio chunks here
          // Feed them into Web Audio API for playback
        }
      } catch (e) {
        console.error("WebSocket message error:", e);
      }
    };

    ws.onclose = () => console.log("🔌 WebSocket disconnected");
    ws.onerror = (err) => console.error("WebSocket error:", err);

    return () => {
      ws.close();
    };
  }, []);

  const getStateColors = () => {
    switch (speakerState) {
      case 1:
        return {
          glow: isDarkMode ? "0 0 80px 12px rgba(0, 153, 255, 0.6)" : "0 0 60px 10px rgba(51, 153, 255, 0.4)",
          gradient: "linear-gradient(135deg, #d0eaff 0%, #66cfff 50%, #3399ff 100%)",
        };
      case 2:
        return {
          glow: isDarkMode ? "0 0 80px 12px rgba(0, 255, 153, 0.5)" : "0 0 60px 10px rgba(0, 204, 102, 0.4)",
          gradient: "linear-gradient(135deg, #ccffe0 0%, #66ffb2 50%, #00cc88 100%)",
        };
      default:
        return {
          glow: isDarkMode ? "0 0 80px 12px rgba(255, 140, 0, 0.5)" : "0 0 60px 10px rgba(255, 179, 71, 0.35)",
          gradient: isDarkMode
            ? "linear-gradient(135deg, #ffddb0 0%, #ffb347 50%, #FFA500 100%)"
            : "linear-gradient(135deg, #fff7e6 0%, #ffe0b2 50%, #ffb347 100%)",
        };
    }
  };

  const { glow, gradient } = getStateColors();

  const handleSaveSettings = async () => {
    console.log("Settings Saved:", { persona, systemPrompt, voice, model });
    try {
      await fetch("http://localhost:8000/configure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          system_prompt: systemPrompt,
          model,
          voice,
        }),
      });
      setSettingsOpen(false);
    } catch (err) {
      console.error("❌ Failed to save settings:", err);
    }
  };

  const toggleSettings = () => setSettingsOpen((prev) => !prev);

  return (
    <div
      className={`min-h-screen flex flex-col items-center justify-center relative px-4 ${
        isDarkMode ? "bg-black" : "bg-white"
      }`}
    >
      <div className="absolute top-6 right-6 z-50">
        <Button onClick={toggleSettings} variant="ghost">
          <Settings className="w-6 h-6" />
        </Button>
      </div>

      <AnimatePresence>
        <motion.div
          key="circle"
          className="relative w-40 h-40 rounded-full overflow-hidden"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: audioLevel }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
          style={{ boxShadow: glow }}
        >
          <motion.div
            className="absolute inset-0 bg-[length:200%_200%]"
            style={{ backgroundImage: gradient }}
            animate={{ backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"] }}
            transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
          />

          <motion.div
            className="absolute inset-0 rounded-full overflow-hidden pointer-events-none"
            style={{
              backgroundImage: "url('https://grainy-gradients.vercel.app/noise.svg')",
              backgroundSize: "300% 300%",
              mixBlendMode: "soft-light",
              opacity: 0.95,
              zIndex: 10,
              backgroundPosition: grainPosition,
            }}
          />
        </motion.div>
      </AnimatePresence>

      <div className="absolute bottom-8 flex gap-4">
        <Button
          onClick={() => setMicMuted(!micMuted)}
          variant={micMuted ? "destructive" : "secondary"}
          className="rounded-full px-6"
        >
          <MicOff className="mr-2 h-4 w-4" />
          {micMuted ? "Unmute Mic" : "Mute Mic"}
        </Button>

        <Button
          onClick={() => setOutputMuted(!outputMuted)}
          variant={outputMuted ? "destructive" : "secondary"}
          className="rounded-full px-6"
        >
          <VolumeX className="mr-2 h-4 w-4" />
          {outputMuted ? "Unmute Output" : "Mute Output"}
        </Button>
      </div>

      <AnimatePresence>
        {settingsOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3 }}
            className="fixed inset-0 z-40 flex items-center justify-center"
          >
            <div className="w-[90%] max-w-md rounded-2xl p-6 backdrop-blur-2xl bg-white/20 dark:bg-black/30 border border-white/20 shadow-xl">
              <div className="flex flex-col gap-4">
                <Select value={persona} onValueChange={setPersona}>
                  <label className="text-sm font-medium">Persona</label>
                  <SelectTrigger>
                    <SelectValue placeholder="Select persona" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="base">Base</SelectItem>
                    <SelectItem value="jarvis">J.A.R.V.I.S.</SelectItem>
                    <SelectItem value="josie">J.O.S.I.E.</SelectItem>
                    <SelectItem value="custom">Custom</SelectItem>
                  </SelectContent>
                </Select>

                <label className="text-sm font-medium">System Prompt</label>
                <Textarea
                  className="resize-none"
                  disabled={persona !== "custom"}
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  placeholder="Enter system prompt..."
                />

                <label className="text-sm font-medium">Voice</label>
                <Select value={voice} onValueChange={setVoice}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select voice" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="default">Default</SelectItem>
                    <SelectItem value="fem1">Female 1</SelectItem>
                    <SelectItem value="male1">Male 1</SelectItem>
                  </SelectContent>
                </Select>

                <label className="text-sm font-medium">Model</label>
                <Select value={model} onValueChange={setModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="gpt-4">GPT-4</SelectItem>
                    <SelectItem value="claude">Claude</SelectItem>
                    <SelectItem value="custom-model">Custom Model</SelectItem>
                  </SelectContent>
                </Select>

                <Button className="mt-4" onClick={handleSaveSettings}>
                  Save Settings
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

```

---

# 🚀 Real-Time S2S Frontend Setup

This is the frontend for the Real-Time Speech-to-Speech system. Built with Next.js, Tailwind CSS, Shadcn UI, and Framer Motion for a sleek and minimal J.A.R.V.I.S.-style interface.

---

## ✅ Prerequisites

Make sure you have:

- **Node.js** (v18+ recommended)
- **npm** (comes with Node) or **yarn**

---

## 📦 Install & Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/real-time-s2s.git
cd real-time-s2s

2. Install dependencies

npm install
# or
yarn

3. Install framer-motion

npm install framer-motion
# or
yarn add framer-motion



⸻

🎨 Shadcn UI Setup

Skip if already initialized

npx shadcn@latest init

Recommended answers during init:

Prompt	Answer
Which style?	default
Use Tailwind?	✅ Yes
App directory	app or src/app
Components directory	components

Add UI components

npx shadcn@latest add button
npx shadcn@latest add select
npx shadcn@latest add textarea



⸻

🧠 File Structure Overview

real-time-s2s/
├── app/
│   └── page.js              ← Mounts <SpeechToSpeechUI />
├── components/
│   └── SpeechToSpeechUI.jsx ← Main animated UI
│   └── ui/                  ← Shadcn UI components
├── public/
├── tailwind.config.js
├── package.json



⸻

▶️ Run the App

npm run dev
# or
yarn dev

Visit: http://localhost:3000

⸻

🤖 Tech Stack
	•	Next.js (App Router)
	•	Tailwind CSS
	•	Shadcn/UI
	•	Framer Motion
	•	Lucide Icons

⸻

🧊 Features
	•	Animated orb interaction
	•	Dark/light mode support
	•	Settings modal with:
	•	Model selector
	•	Voice selector
	•	Persona dropdown
	•	Custom system prompt input
	•	Microphone & output mute toggles

⸻

🛠 Trouble?

If you see BuildError or Can't resolve 'framer-motion', just run:

npm install framer-motion



⸻

Built with 💿 by Gökdeniz