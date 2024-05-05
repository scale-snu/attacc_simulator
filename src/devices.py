from src.type import *
from src.model import *
import math
from src.ramulator_wrapper import *


class xPU:

    def __init__(self, name: DeviceType, config, scaling_factor):
        self.name = name
        self.gpu_type = None
        if self.name == DeviceType.GPU:
            self.gpu_type = config['GPUTYPE']

        self.num_xpu = config['NUM_DEVICE']
        self.num_core = config['NUM_CORE']
        self.peak_flops = config['FLOPS_PER_DEVICE']
        self.peak_memory_bandwidth = config['OFF_MEM_BW_PER_DEVICE']
        self.peak_l2_bandwidth = config['L2_MEM_BW_PER_DEVICE']
        self.l1_cache_size = config['L1_CAP_PER_CORE']
        self.l2_cache_size = config['L2_CAP_PER_DEVICE']
        self.max_interface_bandwidth = config['INTERFACE_BW']
        self.aggregate_memory_capacity = config[
            'MEM_CAPACITY_PER_DEVICE'] * self.num_xpu

        self.max_compute_util = scaling_factor['MAX_COMPUTE_UTIL']
        self.max_memory_util = scaling_factor['MAX_OFF_MEM_BW_UTIL']
        self.energy_table = config['ENERGY_TABLE']

        self.table_tiles = {}

    def _get_traffic_for_tile(self, tm, tn, layer: Layer):
        m, n, k, numOp, dbyte = layer.get_infos()
        traffic = [math.ceil(n / tn) * m * k, math.ceil(m / tm) * n * k, m * n]
        traffic = [i * dbyte * numOp for i in traffic]
        return traffic

    def _get_optimal_tile(self, layer: Layer):
        m, n, k, numOp, dbyte = layer.get_infos()
        config = (m, n, k, numOp, dbyte)
        if config in self.table_tiles.keys():
            return self.table_tiles[config]

        trange = [8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512]

        # find L1 tile size
        l1_tm = 0
        l1_tn = 0
        l1_tk = 32
        opt_config = [0, 0]
        min_cost = float('inf')
        for l1_tm in trange:
            for l1_tn in trange:
                l1_tm = min(l1_tm, m)
                l1_tn = min(l1_tn, n)
                required_capacity = (
                    l1_tm + l1_tn) * l1_tk * dbyte + l1_tm * l1_tn * dbyte
                if required_capacity > self.l1_cache_size:
                    continue
                l2_access = sum(self._get_traffic_for_tile(l1_tm, l1_tn, layer))

                ## applying SM underutilization to cost function
                num_threadblock = numOp
                if layer.type == LayerType.FC:
                    num_threadblock = math.ceil(m / l1_tm) * math.ceil(
                        n / l1_tn) * numOp

                tmp = math.ceil(num_threadblock / self.num_core) * self.num_core
                core_utilization = num_threadblock / tmp
                cost = l2_access * pow((1 / core_utilization), 2)
                if cost < min_cost:
                    min_cost = cost
                    opt_config = [l1_tm, l1_tn]

        l1_tm, l1_tn = opt_config

        # find L2 tile size
        ## experimentally found L2 tile_k size
        l2_tk = k / 64

        min_access = float('inf')
        opt_config = [0, 0]
        for l2_tm in [l1_tm * i for i in range(1, int(m / l1_tm) + 1)] + [m]:
            for l2_tn in [l1_tn * i for i in range(1,
                                                   int(n / l1_tn) + 1)] + [n]:
                l2_tm = min(l2_tm, m)
                l2_tn = min(l2_tn, n)
                required_capacity = (
                    l2_tm + l2_tn) * l2_tk * dbyte + l2_tm * l2_tn * dbyte
                if required_capacity > self.l2_cache_size:
                    if l2_tm != l1_tm or l2_tn != l1_tn:
                        continue

                access = math.ceil(m / l2_tm) * n * k * dbyte + \
                          math.ceil(n / l2_tn) * m * k * dbyte + m * n * dbyte

                if access < min_access:
                    min_access = access
                    opt_config = [l2_tm, l2_tn]

        l2_tm, l2_tn = opt_config
        out_tiles = [l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk]
        self.table_tiles[config] = out_tiles
        return out_tiles

    def _get_traffic(self, layer: Layer):
        # return tuple of 4 elements (off-mem, L2, L1, reg)
        m, n, k, numOp, dbyte = layer.get_infos()
        if layer.type in [
                LayerType.SOFTMAX, LayerType.ACT, LayerType.NORM, LayerType.G2G,
                LayerType.X2G
        ]:
            data = layer.get_size()
            return data, data, data, data

        elif layer.type in [LayerType.FC, LayerType.MATMUL]:
            l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk = self._get_optimal_tile(
                layer)
            reg_tm, reg_tn, reg_tk = 16, 16, 32

            off_data = self._get_traffic_for_tile(l2_tm, l2_tn, layer)
            l2_data = self._get_traffic_for_tile(l1_tm, l1_tn, layer)
            l1_data = self._get_traffic_for_tile(reg_tm, reg_tn, layer)
            reg_data = [m * n * k, m * n * k, m * n * k]

            return off_data, l2_data, l1_data, reg_data

        else:
            assert 0, "Invalid layer type"

    def _compute_time(self, layer: Layer):
        l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk = self._get_optimal_tile(layer)
        m, n, k, numOp, dbyte = layer.get_infos()
        flops = self.peak_flops * self.max_compute_util
        if self.name == DeviceType.GPU:
            num_threadblock = numOp
            if layer.type == LayerType.FC:
                num_threadblock = math.ceil(m / l1_tm) * math.ceil(
                    n / l1_tn) * numOp

            tmp = math.ceil(num_threadblock / self.num_core) * self.num_core
            core_utilization = num_threadblock / tmp

            flops = flops * core_utilization

        ## e.g., peak flops of FP8  is twice that of FP16
        flops *= int(2 / dbyte)

        if flops == 0:
            import pdb
            pdb.set_trace()

        return layer.get_flops() / flops

    def _mem_time(self, layer: Layer):
        l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk = self._get_optimal_tile(layer)
        m, n, k, numOp, dbyte = layer.get_infos()

        off_data, l2_data, l1_data, reg_data = self._get_traffic(layer)
        layer.off_traffic = sum(off_data)

        mem_bw = self.peak_memory_bandwidth * self.max_memory_util
        if self.name == DeviceType.GPU:
            if layer.type == LayerType.ACT:
                exec_time = (
                    0.000000447 *
                    (1555 * 1000 * 1000 * 1000 / self.peak_memory_bandwidth) *
                    sum(off_data) + 8.29) / 1000 / 1000
                return exec_time, 0, 0, 0

            elif layer.type == LayerType.NORM:
                exec_time = (
                    0.0000016 *
                    (1555 * 1000 * 1000 * 1000 / self.peak_memory_bandwidth) *
                    sum(off_data) + 6.87) / 1000 / 1000
                return exec_time, 0, 0, 0

            else:
                num_threadblock = numOp
                if layer.type == LayerType.FC:
                    num_threadblock = math.ceil(m / l1_tm) * math.ceil(
                        n / l1_tn) * numOp

                tmp = math.ceil(num_threadblock / self.num_core) * self.num_core
                core_utilization = num_threadblock / tmp
                mem_bw = mem_bw * core_utilization

                return sum(off_data) / mem_bw, sum(
                    l2_data) / self.peak_l2_bandwidth, 0, 0
        else:
            return sum(off_data) / mem_bw, sum(
                l2_data) / self.peak_l2_bandwidth, 0, 0

    def _exec_time(self, layer: Layer):
        compute_time = self._compute_time(layer)
        mem_time = max(*self._mem_time(layer))
        max_time = 0
        if compute_time > mem_time:
            max_time = compute_time
            layer.bound = "compute"
        else:
            max_time = mem_time
            layer.bound = "memory"
        layer.time = max_time

        return max_time

    def _get_energy(self, layer: Layer):
        off_data, l2_data, l1_data, reg_data = self._get_traffic(layer)
        if self.name == DeviceType.CPU:
            energy_per_acc = self.energy_table['mem']
            e_off = sum(off_data) * energy_per_acc
            e_flop = layer.get_flops() / 2 * self.energy_table['alu']
            energies = [e_off, 0, 0, 0, e_flop, 0]
        else:
            e_off = sum(off_data) * self.energy_table['mem']
            e_l2 = sum(l2_data) * self.energy_table['l2']
            e_l1 = sum(l1_data) * self.energy_table['l1']
            e_reg = sum(reg_data) * self.energy_table['reg']
            e_flop = layer.get_flops() / 2 * self.energy_table['alu']
            energies = [e_off, e_l2, e_l1, e_reg, e_flop, 0]
        energies = [i * self.num_xpu for i in energies]
        return energies

    def _io_time_energy(self, layer: Layer):
        m, n, k, numOp, dbyte = layer.get_infos()

        def get_nvlink_time(size):
            # interpolation of real data on A100
            # size unit: Byte
            if size == 0:
                return 1
            else:
                approx_ns_time = 6060 + 0.009 * size * (
                    (600 * 1000 * 1000 * 1000 / self.max_interface_bandwidth))
                approx_time = approx_ns_time / 1000 / 1000 / 1000
                return max(approx_time,
                           size / (self.max_interface_bandwidth / 2))

        if self.name == DeviceType.CPU:
            ## RX, TX --> 1/2x
            bw = self.max_interface_bandwidth / 2
            traffic = m * n * numOp * dbyte
            exec_time = traffic / bw
            # we ignore CPU energy
            energy = 0
        else:
            ## each GPU has partial sum of output.
            traffic = m * n * numOp * dbyte
            interface_bw = self.max_interface_bandwidth / 2
            if layer.type == LayerType.X2G:
                exec_time = traffic / interface_bw
            else:
                ## allreduce
                exec_time = get_nvlink_time(
                    traffic / self.num_xpu) * (self.num_xpu - 1)

            # all reduce communication
            energy = self.num_xpu * traffic * self.energy_table['comm']
        return exec_time, [0, 0, 0, 0, 0, energy]

    def get_time_and_energy(self, layer: Layer):
        if layer.type in [LayerType.X2G, LayerType.G2G]:
            return self._io_time_energy(layer)
        else:
            return self._exec_time(layer), self._get_energy(layer)


class PIM:

    def __init__(self, config, scaling_factor, ramulator):
        self.name = DeviceType.PIM
        self.num_attacc = config['NUM_ATTACC']
        self.num_hbm = config['NUM_HBM']
        self.pim_type = config['PIM_TYPE']
        self.peak_memory_bandwidth = config['MEM_BW_PER_HBM'] * self.num_hbm
        self.softmax_peak_flops = config['SOFTMAX_FLOPS']
        self.softmax_peak_bandwidth = config['SOFTMAX_MEM_BW']
        self.max_interface_bandwidth = config['INTERFACE_BW']
        self.aggregate_memory_capacity = config[
            'MEM_CAPACITY_PER_HBM'] * self.num_attacc * self.num_hbm
        self.energy_table = config['ENERGY_TABLE']
        self.io_energy_table = self.energy_table['io']
        self.power_constraint = config['POWER_CONSTRAINT']
        self.ramulator = ramulator

    def _get_traffic(self, layer: Layer):
        # return tuple of 4 elements (off-mem, L2, L1, reg)
        m, n, k, numOp, dbyte = layer.get_infos()
        if layer.type in [LayerType.MATMUL, LayerType.FC, LayerType.SOFTMAX]:
            data = layer.get_size()
            return data, [0], [0], [0]

        else:
            assert 0, "In get_traffic function, PIM could not support this layer"

    def _io_time_energy(self, layer: Layer):
        m, n, k, numOp, dbyte = layer.get_infos()
        interface_bw = self.max_interface_bandwidth / 2
        traffic = m * n * numOp * dbyte
        exec_time = traffic / interface_bw

        energy = traffic * self.energy_table['comm'] * self.num_attacc

        return exec_time, [0, 0, 0, 0, 0, energy]

    def _compute_time(self, layer: Layer):
        flops = self.softmax_peak_flops
        flops *= int(2 / layer.dbyte)
        compute_time = layer.get_flops() / flops
        return compute_time

    def _mem_time(self, layer: Layer):
        mem_bw = self.softmax_peak_bandwidth
        mem_time = sum(layer.get_size()) / mem_bw
        return mem_time

    def _get_energy(self, layer: Layer):
        off_data = layer.get_size()
        e_off = sum(off_data) * self.energy_table['sram'] * self.num_attacc
        e_flop = layer.get_flops(
        ) / 2 * self.energy_table['alu'] * self.num_attacc

        return [e_off, 0, 0, 0, e_flop, 0]

    def get_time_and_energy(self, layer: Layer):
        if layer.type == LayerType.X2G:
            return self._io_time_energy(layer)

        elif layer.type == LayerType.MATMUL:
            ## operational granularity = the attention layer
            if 'score' in layer.name:
                m, n, k, numOp, dbyte = layer.get_infos()
                time, traffic = self.ramulator.output(
                    self.pim_type, layer, self.power_constraint)
                io_energy = 0
                for i in range(len(self.io_energy_table)):
                    io_energy += traffic[i] * self.io_energy_table[i]

                energy_per_access = self.energy_table['mem']
                cell_energy = traffic[-1] * energy_per_access
                dram_energy = cell_energy + io_energy
                cal_energy = layer.get_flops() / 2 * self.energy_table['alu']

                energies = [dram_energy, 0, 0, 0, cal_energy, 0]
                energies = [i * self.num_attacc for i in energies]

                return time, energies
            else:
                return 0, [0, 0, 0, 0, 0, 0]

        elif layer.type == LayerType.SOFTMAX:
            # Execution time
            compute_time = self._compute_time(layer)
            mem_time = self._mem_time(layer)

            if compute_time > mem_time:
                layer.bound = 'compute'
            else:
                layer.bound = 'memory'
            exec_time = max(compute_time, mem_time)
            layer.time = exec_time

            energy = self._get_energy(layer)

            return exec_time, energy

        else:
            assert 0, "PIM does not support this layer."
