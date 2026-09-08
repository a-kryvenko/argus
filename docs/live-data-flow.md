# Live data flow

The live page polls one observation summary for all current measurement cards.
The summary includes all solar wind metrics and Kp/Dst, plus derived trends.
Public `latest` endpoints remain available to API clients; the page does not
request them again for hidden details.

Each history and the collection diagnostics refresh independently through
`useLiveQuery`. The hook handles cancellation, retry every minute, retaining the
last response on errors, and hiding results for an old selected period. A single
page clock updates measurement ages. Changing a history period does not refetch
the current snapshot or collection diagnostics.

Collection, aggregation and forecasting remain independent of page requests.
