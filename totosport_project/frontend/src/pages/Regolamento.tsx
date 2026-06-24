import MarkdownView from '@/components/MarkdownView'
// Copia bundlata di docs/rules/REGOLAMENTO.md (fonte di verità delle regole).
// Va tenuta allineata al file in docs/rules/ — vedi 09_frontend_player.md.
import regolamento from '@/content/regolamento.md?raw'

export default function Regolamento() {
  return (
    <div className="mx-auto max-w-3xl">
      <MarkdownView>{regolamento}</MarkdownView>
    </div>
  )
}
