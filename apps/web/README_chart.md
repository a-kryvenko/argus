# Charts customization

```tsx
<Line 
  type="natural" 
  dataKey="gic_risk" 
  stroke="#f97316" // Main line color
  strokeWidth={3.5}
  dot={false}
  activeDot={{ 
    r: 6, 
    fill: '#f97316',
    stroke: '#fff', 
    strokeWidth: 2 
  }}
/>
```

```tsx
<Tooltip
  cursor={true}
  animationDuration={0} // instant tooltip movement without "smooth" animation
  animationEasing="linear"
  contentStyle={{
    backgroundColor: '#18181b',
    border: '1px solid #3f3f46',
    borderRadius: '8px',
    padding: '10px 14px',
    color: '#e4e4e7',
    fontSize: '13px',
    boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.3)',
  }}
  labelStyle={{ color: '#a1a1aa', fontSize: '12px' }}
  itemStyle={{ color: '#f97316' }}
/>
```