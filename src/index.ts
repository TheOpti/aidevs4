import "dotenv/config";

console.log("Starting the main project file!");

// Example of using an environment variable
const apiKey = process.env.OPENAI_API_KEY;

if (!apiKey || apiKey === "your_key_here") {
  console.warn("⚠️  WARNING: Missing a real OPENAI_API_KEY in the .env file!");
} else {
  console.log("✅ API key loaded successfully.");
}
