import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MicOff, VolumeX, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

const personaColors = {
  base: {
    gradient: 'linear-gradient(135deg, #fff7e6 0%, #ffe0b2 50%, #ffb347 100%)',
    glow: '0 0 60px 10px rgba(255, 179, 71, 0.35)'
  },
  jarvis: {
    gradient: 'linear-gradient(135deg, #cfe8ff 0%, #70b7ff 50%, #007BFF 100%)',
    glow: '0 0 80px 12px rgba(0, 123, 255, 0.5)'
  },
  hal9000: {
    gradient: 'linear-gradient(135deg, #ffcccc 0%, #ff6666 50%, #cc0000 100%)',
    glow: '0 0 80px 12px rgba(204, 0, 0, 0.6)'
  },
  missminutes: {
    gradient: 'linear-gradient(135deg, #ffebb3 0%, #ffc266 50%, #e69500 100%)',
    glow: '0 0 80px 12px rgba(230, 149, 0, 0.5)'
  }
};

export default function SpeechToSpeechUI() {
  const [isReady, setIsReady] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [audioLevel, setAudioLevel] = useState(1);
  const [micMuted, setMicMuted] = useState(false);
  const [outputMuted, setOutputMuted] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [persona, setPersona] = useState("base");
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful AI assistant.");
  const [voice, setVoice] = useState("af_bella");
  const [model, setModel] = useState("gemma3");
  const [grainPosition, setGrainPosition] = useState("0% 0%");
  const [connectionState, setConnectionState] = useState(0);

  const socketRef = useRef(null);

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

  const handleSaveSettings = async () => {
    await fetch("http://localhost:8000/configure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ persona, system_prompt: systemPrompt, voice, model })
    });
    setSettingsOpen(false);
    connectWebSocket();
  };

  const connectWebSocket = () => {
    if (socketRef.current) socketRef.current.close();
    const socket = new WebSocket("ws://localhost:8000/realtime-sts");
    socketRef.current = socket;

    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "state") setConnectionState(msg.value);
      } catch (_) {}
    };

    socket.onerror = (err) => {
      console.error("WebSocket Error:", err);
      setConnectionState(0);
    };
  };

  const getVisuals = () => {
    const personaVisual = personaColors[persona.toLowerCase()] || personaColors.base;
    let brightness = 1;
    if (connectionState === 0) brightness = 0.7;
    else if (connectionState === 2) brightness = 1.3;

    return {
      background: `${personaVisual.gradient}`,
      glow: `${personaVisual.glow}`,
      brightness: brightness
    };
  };

  const { background, glow, brightness } = getVisuals();

  return (
    <div className={`min-h-screen flex flex-col items-center justify-center relative px-4 ${isDarkMode ? 'bg-black' : 'bg-white'}`}> 
      <div className="absolute top-6 right-6 z-50">
        <Button onClick={() => setSettingsOpen(prev => !prev)} variant="ghost">
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
          style={{ boxShadow: glow, filter: `brightness(${brightness})` }}
        >
          <motion.div
            className="absolute inset-0 bg-[length:200%_200%]"
            style={{ backgroundImage: background }}
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
                    <SelectItem value="hal9000">Hal9000</SelectItem>
                    <SelectItem value="missminutes">Miss Minutes</SelectItem>
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
                    <SelectItem value="af_bella">af_bella</SelectItem>
                    <SelectItem value="af_sky">af_sky</SelectItem>
                  </SelectContent>
                </Select>

                <label className="text-sm font-medium">Model</label>
                <Select value={model} onValueChange={setModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="gemma3">gemma3</SelectItem>
                    <SelectItem value="gemma3:12b">gemma3:12b</SelectItem>
                    <SelectItem value="qwen2.5">qwen2.5</SelectItem>
                    <SelectItem value="qwen2.5:3b">qwen2.5:3b</SelectItem>
                  </SelectContent>
                </Select>

                <Button className="mt-4" onClick={handleSaveSettings}>
                  Save Settings & Connect
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}