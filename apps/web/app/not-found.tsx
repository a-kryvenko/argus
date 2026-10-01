import Link from 'next/link';
import WorkspaceShell from './_components/WorkspaceShell';
import styles from './_components/forecast.module.css';

export default function NotFound() {
  return <WorkspaceShell section="Page not found" contentId="not-found-content" status="404">
    <main id="not-found-content" className={styles.page}>
      <div className={styles.pageHeading}><div><div className={styles.eyebrow}>404 / PAGE NOT FOUND</div><h1>Page not found</h1><p>This address does not match an available page.</p></div></div>
      <Link className={styles.headingLink} href="/">Return to forecast overview →</Link>
    </main>
  </WorkspaceShell>;
}
