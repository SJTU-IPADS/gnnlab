#pragma once

#include <cstdio>
#include <vector>

struct Time {
    double last = 0;
    size_t num_times = 0;
    double sum = 0;
    double avg = 0;

    void Log(double time) {
        last = time;
        num_times++;
        sum += time;
        avg = sum / num_times;
    }
};

struct BlockTime {
    Time block;
    Time sample;
    Time remap;
    Time create_coo;
    Time coo2csr;
    Time csr2csc;
};

class Performance {
public:
    Time epochs;
    Time shuffles;
    Time steps;
    Time sampling;
    std::vector<BlockTime> blocks;
    Time index_select;

    static Performance& Instance() {
        static Performance instance;
        return instance;
    }

    void SetMeta(int num_blocks) {
        this->num_blocks = num_blocks;
        this->blocks.resize(num_blocks);
    }

    void ReportStep(int epoch, int step) {
        // printf("Epoch %d, step %d, total %.4lfs/%.4lfs/%.2lf%%, prepare %.4lfs/%.4lfs/%.2lf%%, index_select %.4lfs/%.4lfs/%.2lf%%\n",
        //        epoch, step,
        //        steps.last, steps.avg, steps.last / steps.last * 100,
        //        prepare.last, prepare.avg, prepare.last / steps.last * 100,
        //        index_select.last, index_select.avg, index_select.last / steps.last * 100);

        // for (int bid = 0; bid < num_blocks; bid++) {
        //     printf("  Block %d, total %.4lfs/%.4lfs/%.2lf%%, sample %.4lfs/%.4lfs/%.2lf%%, remap %.4lfs/%.4lfs/%.2lf%%, coo %.4lfs/%.4lfs/%.2lf%%, csr %.4lfs/%.4lfs/%.2lf%%, csc %.4lfs/%.4lfs/%.2lf%%\n",
        //            bid,
        //            blocks[bid].block.last, blocks[bid].block.avg, blocks[bid].block.last / steps.last * 100,
        //            blocks[bid].sample.last, blocks[bid].sample.avg, blocks[bid].sample.last / steps.last * 100,
        //            blocks[bid].remap.last, blocks[bid].remap.avg, blocks[bid].remap.last / steps.last * 100,
        //            blocks[bid].create_coo.last, blocks[bid].create_coo.avg, blocks[bid].create_coo.last / steps.last * 100,
        //            blocks[bid].coo2csr.last, blocks[bid].coo2csr.avg, blocks[bid].coo2csr.last / steps.last * 100,
        //            blocks[bid].csr2csc.last, blocks[bid].csr2csc.avg, blocks[bid].csr2csc.last / steps.last * 100
        //            );
        // }
        printf("Epoch %d, step %d, total %.2lf%%, sampling %.2lf%%, index_select %.2lf%%\n",
               epoch, step,
               steps.last / steps.last * 100,
               sampling.last / steps.last * 100,
               index_select.last / steps.last * 100);

        for (int bid = 0; bid < num_blocks; bid++) {
            printf("  Block %d, total %.2lf%%, sample %.2lf%%, remap %.2lf%%, coo %.2lf%%, csr %.2lf%%, csc %.2lf%%\n",
                   bid,
                   blocks[bid].block.last / steps.last * 100,
                   blocks[bid].sample.last / steps.last * 100,
                   blocks[bid].remap.last / steps.last * 100,
                   blocks[bid].create_coo.last / steps.last * 100,
                   blocks[bid].coo2csr.last / steps.last * 100,
                   blocks[bid].csr2csc.last / steps.last * 100
                   );
        }
    }

    void ReportEpoch() {

    }

private:
    int num_blocks = 0;
};
