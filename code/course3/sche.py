import random
random.seed(42)

BLOCK_SIZE = 4
TOKEN_BUDGET = 10
MAX_BLOCKS = 16          # 模拟 GPU 显存上限


# ------------------ 基本数据结构 ------------------
class Request:
    _id_counter = 0

    def __init__(self, prompt_tokens: int, decode_tokens: int = 8):
        Request._id_counter += 1
        self.request_id = f"req-{Request._id_counter}"
        self.prompt_len = prompt_tokens
        self.decode_tokens = decode_tokens           # 还需生成的 token 数
        self.num_computed_tokens = 0                 # 已算过的 token
        self.status = "WAITING"                      # WAITING / RUNNING / DONE

    def __repr__(self):
        return (f"{self.request_id}({self.status}, "
                f"computed={self.num_computed_tokens}/{self.prompt_len}, "
                f"remain_decode={self.decode_tokens})")


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.tokens = 0
        self.free = True


# ------------------ 极简 KVCacheManager ------------------
class KVCacheManager:
    def __init__(self, max_blocks: int):
        self.blocks = [Block(i) for i in range(max_blocks)]
        self.used = 0

    def _alloc_blocks(self, num_tokens: int) -> list[Block] | None:
        need = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        if self.used + need > len(self.blocks):
            return None
        allocated = []
        for b in self.blocks:
            if b.free:
                b.free = False
                allocated.append(b)
                self.used += 1
                if len(allocated) == need:
                    return allocated
        return None

    def free_blocks(self, blocks: list[Block]):
        for b in blocks:
            b.free = True
            self.used -= 1


# ------------------ 调度器 ------------------
class Scheduler:
    def __init__(self):
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.kv_mgr = KVCacheManager(MAX_BLOCKS)
        self.step_no = 0

    def add_request(self, req: Request):
        self.waiting.append(req)

    def _try_schedule_one(self, req: Request, token_budget: int) -> int | None:
        """返回实际可调度 token 数，若资源不足返回 None"""
        need = min(req.prompt_len - req.num_computed_tokens, token_budget)
        if need == 0:
            return 0
        blocks = self.kv_mgr._alloc_blocks(need)
        if blocks is None:
            return None
        return need

    def step(self):
        self.step_no += 1
        print(f"\n========== step {self.step_no} ==========")
        token_budget = TOKEN_BUDGET
        scheduled = []           # (req, tokens)
        preempted = []

        # 1. 先调度 running（decode 阶段）
        for req in self.running.copy():
            if token_budget <= 0:
                break
            # decode 阶段每次算 1 个 token
            need = 1
            blocks = self.kv_mgr._alloc_blocks(need)
            if blocks is None:
                # 抢占自己
                preempted.append(req)
                self.running.remove(req)
                continue
            scheduled.append((req, need))
            token_budget -= need
            req.decode_tokens -= 1
            req.num_computed_tokens += 1
            if req.decode_tokens == 0:
                req.status = "DONE"

        # 2. 再调度 waiting（prefill）
        while self.waiting and token_budget > 0 and len(preempted) == 0:
            req = self.waiting[0]
            need = self._try_schedule_one(req, token_budget)
            if need is None:
                # 资源不足，抢占 running 末尾
                if not self.running:
                    break
                victim = self.running.pop()
                preempted.append(victim)
                print(f"  preempt {victim.request_id}")
                victim.status = "WAITING"
                victim.num_computed_tokens = 0
                self.waiting.insert(0, victim)
                continue
            # 成功调度
            self.waiting.pop(0)
            self.running.append(req)
            req.status = "RUNNING"
            scheduled.append((req, need))
            token_budget -= need
            req.num_computed_tokens += need

        # 3. 打印决策
        for req, tok in scheduled:
            print(f"  schedule {req.request_id}  tokens={tok}")
        print("  waiting :", self.waiting)
        print("  running :", self.running)
        return len(scheduled) > 0 or preempted


# ------------------ 驱动 ------------------
def main():
    sched = Scheduler()
    # 模拟 6 个请求随机到达
    for _ in range(6):
        sched.add_request(Request(prompt_tokens=random.randint(4, 12),
                          decode_tokens=random.randint(3, 7)))

    # 循环调度直到全部完成
    while sched.waiting or sched.running:
        made_progress = sched.step()
        if not made_progress:
            print("  no progress, break")
            break
    print("\nAll done!")


if __name__ == "__main__":
    main()