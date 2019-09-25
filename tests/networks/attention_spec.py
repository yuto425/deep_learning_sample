import torch
import random
from mamba import description, context, it, before
from expects import expect, equal
from networks.attention import SimpleAttention


with description('SimpleAttention') as self:
    with before.each:
        self.batch_size = 2
        self.depth = 300
        self.input_length = 5
        self.memory_length = 3
        self.input = torch.randn(
            self.batch_size, self.input_length, self.depth,
            dtype=torch.float32)
        self.memory = torch.randn(
            self.batch_size, self.memory_length, self.depth,
            dtype=torch.float32)
        self.mask = torch.tensor(
            [[[1 if random.random() * self.memory_length > i else 0
               for i in range(self.memory_length)]
              for _ in range(self.input_length)]
             for _ in range(self.batch_size)],
            dtype=torch.bool)

    with context('when calculate with input, memory, and mask'):
        with it('should return a tensor with expected shape'):
            model = SimpleAttention(self.depth)
            output = model(self.input, self.memory, self.mask)
            expect(list(output.size())).to(equal(
                [self.batch_size, self.input_length, self.depth]))
