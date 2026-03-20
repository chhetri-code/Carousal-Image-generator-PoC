| Area       | Action                                             |
| ---------- | -------------------------------------------------- |
| Env        | Remove `.env` → use `st.secrets`                   |
| LLM        | Remove LangChain → direct Groq API call            |
| Prompt     | Include business name, website, location, theme    |
| Caption    | Generate one caption per carousel                  |
| CSS        | Reduce prompt/enhanced prompt font size            |
| Images     | Replace deprecated `use_column_width` with `width` |
| Download   | 4-column layout + ZIP download                     |
| Footer     | Add 🇮🇳 Made in India with ❤️ by Chhetri          |
| Validation | Warn if mandatory inputs missing                   |
