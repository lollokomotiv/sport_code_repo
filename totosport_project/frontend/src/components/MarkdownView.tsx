import ReactMarkdown from 'react-markdown'
import rehypeSlug from 'rehype-slug'
import remarkGfm from 'remark-gfm'

/**
 * Rende testo markdown (con tabelle GFM) applicando lo stile `.markdown`.
 * `rehype-slug` aggiunge gli `id` ai titoli, così i link dell'indice (`#...`)
 * possono saltare alla sezione corrispondente.
 */
export default function MarkdownView({ children }: { children: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSlug]}>
        {children}
      </ReactMarkdown>
    </div>
  )
}
