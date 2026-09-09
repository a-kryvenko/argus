export default function ResourceState({
  error,
  retry,
  label,
}: {
  error: string | null;
  retry: () => void;
  label: string;
}) {
  return (
    <div className="state-message" role={error ? "alert" : "status"}>
      <p>{error ?? `Loading ${label}…`}</p>
      {error && (
        <button className="retry-button" onClick={retry}>
          Try again
        </button>
      )}
    </div>
  );
}
