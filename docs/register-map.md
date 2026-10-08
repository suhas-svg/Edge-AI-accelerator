# Register map v0.1 (frozen with hardware partner before RTL)

| Register | Offset | Description |
| --- | --- | --- |
| CONTROL | 0x0000 | Start/reset/control |
| STATUS | 0x0004 | Busy/complete/error |
| CMD_ADDR | 0x0008 | Command-buffer address |
| CMD_SIZE | 0x000C | Command-buffer size |
| INPUT_ADDR | 0x0010 | Input buffer |
| OUTPUT_ADDR | 0x0014 | Output buffer |
| IRQ_STATUS | 0x0018 | Interrupt status |
| IRQ_ENABLE | 0x001C | Interrupt enable |
| PERF_CYCLES | 0x0020 | Execution cycles |
| PERF_MACS | 0x0024 | Number of MAC operations |

Status: draft. Needs partner signoff on widths, reset values, and IRQ behavior.
