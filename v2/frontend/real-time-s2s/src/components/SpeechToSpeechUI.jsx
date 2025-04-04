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

  const [speaker, setSpeaker] = useState("none");

  const [persona, setPersona] = useState("base");
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful AI assistant.");
  const [voice, setVoice] = useState("");
  const [model, setModel] = useState("");
  const [availableVoices, setAvailableVoices] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);
  const [grainPosition, setGrainPosition] = useState("0% 0%");

  useEffect(() => {
    const timer = setTimeout(() => setIsReady(true), 2000);
    const match = window.matchMedia("(prefers-color-scheme: dark)");
    setIsDarkMode(match.matches);
    match.addEventListener("change", (e) => setIsDarkMode(e.matches));
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const socket = new WebSocket("ws://127.0.0.1:8000/ws/state");
    socket.onmessage = (event) => {
      const state = event.data;
      if (state === "user") {
        setSpeaker("user");
        setAudioLevel(1.2);
      } else if (state === "assistant") {
        setSpeaker("assistant");
        setAudioLevel(1.4);
      } else {
        setSpeaker("none");
        setAudioLevel(1);
      }
    };
    return () => socket.close();
  }, []);

  useEffect(() => {
    let retryCount = 0;
    const maxRetries = 5;
    const retryDelay = 5000;

    const fetchConfig = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/get-config");
        const data = await res.json();
        setSystemPrompt(data.system_prompt || "");
        setModel(data.model || "");
        setVoice(data.voice || "");
        setPersona(data.persona || "base");
      } catch (err) {
        retryCount++;
        if (retryCount < maxRetries) {
          console.warn(`Connection failed. Retrying (${retryCount}/${maxRetries})...`);
          setTimeout(fetchConfig, retryDelay);
        } else {
          alert("⚠️ Trouble connecting to the backend. Please restart the service.");
        }
      }
    };

    fetchConfig();
  }, []);

  useEffect(() => {
    const fetchVoicesAndModels = async () => {
      try {
        const voicesRes = await fetch("http://localhost:8880/v1/audio/voices");
        const voicesData = await voicesRes.json();
        setAvailableVoices(Array.isArray(voicesData?.voices) ? voicesData.voices : []);

        const modelsRes = await fetch("http://localhost:11434/api/tags");
        const modelsData = await modelsRes.json();
        setAvailableModels(Array.isArray(modelsData?.models) ? modelsData.models.map(m => m.name) : []);
      } catch (err) {
        console.error("Error fetching voices/models", err);
      }
    };

    fetchVoicesAndModels();
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

  const gradientLight = 'linear-gradient(135deg, #fff7e6 0%, #ffe0b2 50%, #ffb347 100%)';
  const gradientDark = 'linear-gradient(135deg, #ffddb0 0%, #ffb347 50%, #FFA500 100%)';
  const glowLight = '0 0 60px 10px rgba(255, 179, 71, 0.35)';
  const glowDark = '0 0 80px 12px rgba(255, 140, 0, 0.5)';

  const handleSaveSettings = async () => {
    await fetch("http://localhost:8000/update-config", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        persona,
        system_prompt: persona === "custom" ? systemPrompt : "",
        voice,
        model,
      }),
    });
    setSettingsOpen(false);
  };

  const toggleSettings = async () => {
    setSettingsOpen(prev => !prev);

    if (!settingsOpen) {
      try {
        const res = await fetch("http://127.0.0.1:8000/get-config");
        const data = await res.json();
        setSystemPrompt(data.system_prompt || "");
        setModel(data.model || "");
        setVoice(data.voice || "");
        setPersona(data.persona || "base");
      } catch (err) {
        console.error("Failed to fetch config", err);
      }
    }
  };

  return (
    <div className={`min-h-screen flex flex-col items-center justify-center relative px-4 ${isDarkMode ? 'bg-black' : 'bg-white'}`}> 
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
          animate={{ 
            opacity: 1,
            scale: 1 + (audioLevel - 1) * 1.1  // exaggerates fluctuation around 1
          }}
          transition={{ type: "spring", stiffness: 500, damping: 25 }}
          style={{ boxShadow: isDarkMode ? glowDark : glowLight }}
        >
          <motion.div
            className="absolute inset-0 bg-[length:200%_200%]"
            style={{ backgroundImage: isDarkMode ? gradientDark : gradientLight }}
            animate={{ backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"] }}
            transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
          />

          <div
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
          onClick={async () => {
            const endpoint = micMuted ? "/unmute-mic" : "/mute-mic";
            try {
              const res = await fetch(`http://127.0.0.1:8000${endpoint}`);
              if (res.ok) setMicMuted(!micMuted);
            } catch (err) {
              console.error("Mic toggle failed", err);
            }
          }}
          variant={micMuted ? "destructive" : "secondary"}
          className="rounded-full px-6"
        >
          <MicOff className="mr-2 h-4 w-4" />
          {micMuted ? "Unmute Mic" : "Mute Mic"}
        </Button>

        <Button
          onClick={async () => {
            const endpoint = outputMuted ? "/unmute-assistant" : "/mute-assistant";
            try {
              const res = await fetch(`http://127.0.0.1:8000${endpoint}`);
              if (res.ok) setOutputMuted(!outputMuted);
            } catch (err) {
              console.error("Assistant output toggle failed", err);
            }
          }}
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
                    <SelectItem value="miss_minutes">Miss Minutes</SelectItem>
                    <SelectItem value="hall9000">Hall900</SelectItem>
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
                    {availableVoices.map((v) => (
                      <SelectItem key={v} value={v}>{v}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <label className="text-sm font-medium">Model</label>
                <Select value={model} onValueChange={setModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableModels.map((m) => (
                      <SelectItem key={m} value={m}>{m}</SelectItem>
                    ))}
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
