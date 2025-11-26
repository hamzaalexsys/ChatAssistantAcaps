import { useState, useRef, useEffect } from 'react'
import { Send, FileText, ExternalLink, Bot, User, Sparkles, AlertCircle, RefreshCw } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

// Types
interface Citation {
  title: string
  url: string
  score: number
  snippet: string
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  confidence?: number
  timestamp: Date
  isLoading?: boolean
}

// API Service
const API_BASE = '/api'

async function sendQuery(question: string, conversationId?: string): Promise<{
  answer: string
  citations: Citation[]
  confidence: number
  conversation_id: string
}> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      conversation_id: conversationId,
    }),
  })
  
  if (!response.ok) {
    throw new Error('Failed to get response')
  }
  
  return response.json()
}

// Components
function TypingIndicator() {
  return (
    <div className="flex items-center space-x-1 px-4 py-2">
      <div className="typing-dot" />
      <div className="typing-dot" />
      <div className="typing-dot" />
    </div>
  )
}

function CitationCard({ citation }: { citation: Citation }) {
  return (
    <a
      href={citation.url}
      target="_blank"
      rel="noopener noreferrer"
      className="citation-card block p-3 bg-white border border-slate-200 rounded-lg hover:border-acaps-accent"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-medium text-acaps-primary">
          <FileText className="w-4 h-4 text-acaps-accent" />
          <span className="line-clamp-1">{citation.title}</span>
        </div>
        <ExternalLink className="w-4 h-4 text-slate-400 flex-shrink-0" />
      </div>
      <p className="mt-2 text-xs text-slate-500 line-clamp-2">{citation.snippet}</p>
      <div className="mt-2 flex items-center gap-1">
        <div className="h-1 flex-1 bg-slate-100 rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-acaps-accent to-acaps-secondary rounded-full"
            style={{ width: `${citation.score * 100}%` }}
          />
        </div>
        <span className="text-xs text-slate-400">{Math.round(citation.score * 100)}%</span>
      </div>
    </a>
  )
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  
  return (
    <div className={`flex gap-3 message-enter ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div className={`flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center ${
        isUser 
          ? 'bg-gradient-to-br from-acaps-primary to-acaps-secondary' 
          : 'bg-gradient-to-br from-acaps-accent to-blue-400'
      }`}>
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : (
          <Bot className="w-5 h-5 text-white" />
        )}
      </div>
      
      {/* Content */}
      <div className={`flex-1 max-w-[80%] ${isUser ? 'text-right' : ''}`}>
        <div className={`inline-block rounded-2xl px-4 py-3 ${
          isUser 
            ? 'bg-acaps-primary text-white rounded-tr-sm' 
            : 'bg-white border border-slate-200 text-slate-700 rounded-tl-sm shadow-sm'
        }`}>
          {message.isLoading ? (
            <TypingIndicator />
          ) : (
            <div className="prose prose-sm max-w-none prose-slate">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>
        
        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div className="mt-3 space-y-2">
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Sparkles className="w-3 h-3" />
              <span>Sources ({message.citations.length})</span>
            </div>
            <div className="grid gap-2">
              {message.citations.map((citation, idx) => (
                <CitationCard key={idx} citation={citation} />
              ))}
            </div>
          </div>
        )}
        
        {/* Timestamp */}
        <div className={`mt-1 text-xs text-slate-400 ${isUser ? 'text-right' : ''}`}>
          {message.timestamp.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>
    </div>
  )
}

function SuggestionChip({ text, onClick }: { text: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-4 py-2 bg-white border border-slate-200 rounded-full text-sm text-slate-600 hover:border-acaps-accent hover:text-acaps-primary transition-colors"
    >
      {text}
    </button>
  )
}

// Main App
export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [conversationId, setConversationId] = useState<string | undefined>()
  const [error, setError] = useState<string | null>(null)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  
  const suggestions = [
    "Que dit l'Article 5 sur les congés maladie ?",
    "Quels sont les horaires de travail ?",
    "Comment demander des congés annuels ?",
    "Quelles sont les obligations de confidentialité ?"
  ]
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }
  
  useEffect(() => {
    scrollToBottom()
  }, [messages])
  
  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault()
    
    if (!input.trim() || isLoading) return
    
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    }
    
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setError(null)
    
    // Add loading message
    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true
    }
    setMessages(prev => [...prev, loadingMessage])
    
    try {
      const response = await sendQuery(userMessage.content, conversationId)
      setConversationId(response.conversation_id)
      
      // Replace loading message with actual response
      const assistantMessage: Message = {
        id: loadingMessage.id,
        role: 'assistant',
        content: response.answer,
        citations: response.citations,
        confidence: response.confidence,
        timestamp: new Date()
      }
      
      setMessages(prev => prev.map(m => m.id === loadingMessage.id ? assistantMessage : m))
    } catch (err) {
      console.error('Error:', err)
      setError('Une erreur est survenue. Veuillez réessayer.')
      // Remove loading message
      setMessages(prev => prev.filter(m => m.id !== loadingMessage.id))
    } finally {
      setIsLoading(false)
    }
  }
  
  const handleSuggestion = (text: string) => {
    setInput(text)
    inputRef.current?.focus()
  }
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }
  
  const handleNewChat = () => {
    setMessages([])
    setConversationId(undefined)
    setError(null)
  }
  
  return (
    <div className="h-screen flex flex-col bg-slate-50 pattern-bg">
      {/* Header */}
      <header className="flex-shrink-0 gradient-bg text-white shadow-lg">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-white/20 backdrop-blur rounded-xl flex items-center justify-center">
                <Bot className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-xl font-display font-bold">Atlas-RAG</h1>
                <p className="text-sm text-blue-200">Assistant Documentation ACAPS</p>
              </div>
            </div>
            <button
              onClick={handleNewChat}
              className="flex items-center gap-2 px-3 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              <span className="text-sm">Nouvelle conversation</span>
            </button>
          </div>
        </div>
      </header>
      
      {/* Messages */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
          {messages.length === 0 ? (
            // Welcome screen
            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-acaps-accent to-acaps-secondary rounded-2xl flex items-center justify-center mb-6 shadow-lg shadow-blue-500/20">
                <Sparkles className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-2xl font-display font-bold text-acaps-primary mb-2">
                Bienvenue sur Atlas-RAG
              </h2>
              <p className="text-slate-500 mb-8 max-w-md">
                Posez vos questions sur les réglementations ACAPS, les procédures internes et l'utilisation du site.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {suggestions.map((suggestion, idx) => (
                  <SuggestionChip 
                    key={idx} 
                    text={suggestion} 
                    onClick={() => handleSuggestion(suggestion)} 
                  />
                ))}
              </div>
            </div>
          ) : (
            messages.map(message => (
              <MessageBubble key={message.id} message={message} />
            ))
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>
      
      {/* Error banner */}
      {error && (
        <div className="bg-red-50 border-t border-red-200 px-4 py-3">
          <div className="max-w-4xl mx-auto flex items-center gap-2 text-red-700">
            <AlertCircle className="w-5 h-5" />
            <span className="text-sm">{error}</span>
          </div>
        </div>
      )}
      
      {/* Input */}
      <footer className="flex-shrink-0 bg-white border-t border-slate-200 shadow-lg">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <form onSubmit={handleSubmit} className="flex items-end gap-3">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Posez votre question..."
                rows={1}
                className="w-full px-4 py-3 pr-12 bg-slate-50 border border-slate-200 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-acaps-accent focus:border-transparent transition-all"
                style={{ minHeight: '48px', maxHeight: '120px' }}
                disabled={isLoading}
              />
            </div>
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="flex-shrink-0 w-12 h-12 bg-gradient-to-r from-acaps-primary to-acaps-secondary text-white rounded-xl flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-blue-500/25 transition-all"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
          <p className="mt-2 text-xs text-center text-slate-400">
            Atlas-RAG répond uniquement sur la base des documents disponibles.
          </p>
        </div>
      </footer>
    </div>
  )
}

