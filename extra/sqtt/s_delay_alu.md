ALU Instruction Software Scheduling
The shader program may include instructions to delay ALU instructions from being issued in order to attempt
to avoid pipeline stalls caused by issuing dependent instructions too closely together.
This is accomplished with the: S_DELAY_ALU instruction: "insert delay with respect to a previous VALU
instruction". The compiler may insert S_DELAY_ALU instructions to indicate data dependencies that might
benefit from having extra idle cycles inserted between them.
This instruction is inserted before the instruction which the user wants to delay, and it specifies which
previous instructions this one is dependent on. The hardware then determines the number of cycles of delay to
add.
This instruction is optional - it is not necessary for correct operation. It should be inserted only when necessary
to avoid dependency stalls. If enough independent instructions are between dependent ones then no delay is
necessary. For wave64, the user may not know the status of the EXEC mask and hence not know if instructions
take 1 or 2 passes to issue.
The S_DELAY_ALU instruction says: wait for the VALU-Inst N ago to have completed. To reduce instruction
stream overhead, the S_DELAY_ALU instructions packs two delay values into one instruction, with a "skip"
indicator so the two delayed instructions don’t need to be back-to-back.
S_DELAY_ALU may be executed in zero cycles - it may be executed in parallel with the instruction before it.
This avoids extra delay if no delay is needed.

INSTID
SKIP
counts backwards N VALU instructions that were issued. This means it does not count
instructions which were branched over. VALU instructions skipped due to EXEC==0 do count
(scoreboard immediately marked 'ready').
counts the number of instructions skipped before the instruction which has the second
dependency. Every instruction is counted for skipping - all types.
If another S_DELAY_ALU is encountered before the info from the previous one is consumed, the current
S_DELAY_ALU replaces any previous dependency info. This means if an instruction is dependent on two
separate previous instructions, both of those dependencies can be expressed in a single S_DELAY_ALU op, but
not in two separate S_DELAY_ALU ops.
S_DELAY_ALU is applied to any type of opcode, even non-alu (but serves no purpose).
S_DELAY_ALU should not be used within VALU clauses.

# Table 19. S_DELAY_ALU Instruction Codes

| DEP Code | Dep Code Meaning                                      | SKIP Code | SKIP Code Meaning |
|----------|-------------------------------------------------------|-----------|-------------------|
| 0        | no dependency                                         | 0         | Same op. Both DEP codes apply to the next instruction |
| 1-4      | dependent on previous VALU<br>1-4 back                | 1         | No skip. Dep0 applies to the following instruction, and Dep1 applies to the instruction after that one. |
| 5-7      | dependent on previous trans.<br>VALU 1-4 back         | 2         | Skip 1. Dep0 applies to the following instruction. Dep1 applies to 2 instructions ahead (skip 1 instruction). |
| 8        | Reserved                                              | 3-5       | Skip 2-4 instructions between Dep0 and Dep1. |
| 9-11     | Wait 1-3 cycles for previous<br>SALU ops              | 6         | Reserved |

**Note:**  
Codes 9-11: SALU ops typically complete in a single cycle, so waiting for 1 cycle is roughly equivalent to waiting for 1 SALU op to execute before continuing.